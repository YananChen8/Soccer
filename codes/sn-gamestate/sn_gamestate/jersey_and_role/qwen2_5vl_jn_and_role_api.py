import pandas as pd
import torch
import numpy as np
import logging
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from tracklab.utils.collate import default_collate, Unbatchable
from tracklab.pipeline.detectionlevel_module import DetectionLevelModule

from multiprocessing import Pool

log = logging.getLogger(__name__)

class QWEN2_5VL_JN_AND_ROLE_BATCH(DetectionLevelModule):
    input_columns = ["bbox_ltwh"]
    output_columns = ["jersey_number_detection", "jersey_number_confidence", "role_detection", "role_confidence"]
    collate_fn = default_collate

    def __init__(self, cfg, batch_size, device, tracking_dataset=None):
        super().__init__(batch_size=batch_size)
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.cfg = cfg
        self.model_path = self.cfg.model_path
        self.save_jersey_number_full_detection = cfg.save_jersey_number_full_detection
        if self.save_jersey_number_full_detection:
            self.output_columns.append("jersey_number_full_detection")
        self.use_legibility_filter = cfg.use_legibility_filter
        if self.use_legibility_filter:
            self.input_columns.append("legibility_score")
            self.legibility_filter_threshold = cfg.legibility_filter_threshold

        # ---- acceleration knobs ------------------------------------------------
        # frame_stride: only run Qwen on every N-th image_id (1 = all frames).
        # Majority-vote tracklet aggregation downstream tolerates sparse labels.
        self.frame_stride = int(getattr(cfg, 'frame_stride', 1))
        # skip_role_inference: skip Qwen role pass entirely; re-use PRTReId's
        # role_detection/role_confidence (already computed earlier in pipeline).
        self.skip_role_inference = bool(getattr(cfg, 'skip_role_inference', False))
        # ------------------------------------------------------------------------

        if '2.5' in self.model_path:
            model_api = Qwen2_5_VLForConditionalGeneration
        elif '3' in self.model_path:
            model_api = Qwen3VLMoeForConditionalGeneration
        else:
            raise ValueError(f"Model path {self.model_path} is not supported")

        self.model = model_api.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            max_pixels=int(cfg.image_width * cfg.image_height / cfg.downsample_factor / cfg.downsample_factor) // (28 * 28) * (28 * 28),
            use_fast=False,
        )
        self.processor.tokenizer.padding_side = "left"
        self.batch_size = batch_size
        self.device = device

        # Jersey number detection prompt
        self.jn_text_prompt = "Analyze this image and determine if the player is facing away from the camera. If the player is facing away, output the jersey number on their back. If the player is not facing away from the camera, output 'No'."

        # Role detection prompt
        self.role_text_prompt = """
            There are an image and a crop from it.
            Analyze this image and determine the role of the person in this image.
            Respond ONLY with a single word in ['player', 'referee', 'goalkeeper', 'other'].
            If there is no person in the image, or the person is not an athlete on the pitch, respond 'other'.
            """
        self.role_list = ['player', 'referee', 'goalkeeper', 'other']

        self.NUM_BEAMS = 1
        self.TEMPERATURE = 0.0
        self.MAX_NEW_TOKENS = 128
        self.USE_CACHE = True
        self.DO_SAMPLE = True if self.TEMPERATURE > 0 else False

        log.info(
            f"[QWEN FAST] frame_stride={self.frame_stride}  "
            f"skip_role_inference={self.skip_role_inference}  "
            f"batch_size={self.batch_size}"
        )

    def no_jersey_number(self):
        return None, 0

    @torch.no_grad()
    def preprocess(self, image, detection: pd.Series, metadata: pd.Series):
        l, t, r, b = detection.bbox.ltrb(
            image_shape=(image.shape[1], image.shape[0]), rounded=True
        )
        crop = image[t:b, l:r]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            crop = np.zeros((10, 10, 3), dtype=np.uint8)

        batch = {
            'images': Unbatchable([image]),
            'crops': Unbatchable([crop]),
        }

        if self.use_legibility_filter:
            batch['legibility_score'] = detection.legibility_score

        return batch

    def extract_numbers(self, text):
        if text.strip() == "?":
            return None
        number = ''
        for char in text:
            if char.isdigit():
                number += char
        return number if number != '' else None

    def extract_role(self, text):
        text = text.lower()
        if text in self.role_list:
            return text
        return None

    # ------------------------------------------------------------------
    # Helper: build a boolean stride mask for the current batch.
    # Returns a list of length real_bs; True = process with Qwen.
    # ------------------------------------------------------------------
    def _stride_mask(self, real_bs: int, detections: pd.DataFrame):
        if self.frame_stride <= 1:
            return [True] * real_bs

        # Prefer image_id → frame → fallback to position-in-batch
        img_id_col = next(
            (c for c in ('image_id', 'frame_id', 'frame') if c in detections.columns),
            None,
        )
        if img_id_col is not None:
            return [int(v) % self.frame_stride == 0
                    for v in detections[img_id_col].values]
        else:
            # Fallback: keep every frame_stride-th detection by batch position
            log.warning(
                "[QWEN FAST] image_id column not found; "
                "falling back to positional stride"
            )
            return [i % self.frame_stride == 0 for i in range(real_bs)]

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        real_bs = len(batch['crops'])

        # Initialise output lists with "no prediction" defaults
        jersey_number_detection  = [None]  * real_bs
        jersey_number_confidence = [0.0]   * real_bs
        jersey_number_full_detection = [''] * real_bs
        role_detection           = [None]  * real_bs
        role_confidence          = [0.0]   * real_bs

        # ── temporal subsampling mask ────────────────────────────────────
        stride_mask = self._stride_mask(real_bs, detections)
        n_selected  = sum(stride_mask)
        log.debug(f"[QWEN FAST] batch={real_bs}  stride_selected={n_selected}")

        # ════════════════════════════════════════════════════════════════
        # Step 1 – Jersey Number (JN) detection
        # Only on stride-selected AND legibility-passing detections.
        # ════════════════════════════════════════════════════════════════
        jn_idxs = []
        if self.use_legibility_filter:
            for i, (score, sel) in enumerate(
                    zip(batch['legibility_score'], stride_mask)):
                if sel and score >= self.legibility_filter_threshold:
                    jn_idxs.append(i)
        else:
            jn_idxs = [i for i, sel in enumerate(stride_mask) if sel]

        if len(jn_idxs) > 0:
            crops = [batch['crops'][idx].cpu().numpy() for idx in jn_idxs]

            messages = [
                [{"role": "user", "content": [
                    {"type": "image", "image": Image.fromarray(crop)},
                    {"type": "text",  "text":  self.jn_text_prompt},
                ]}]
                for crop in crops
            ]
            texts = [
                self.processor.apply_chat_template(msg, tokenize=False,
                                                   add_generation_prompt=True)
                for msg in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=texts, images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(self.device)

            generated_ids = self.model.generate(
                **inputs,
                num_beams=self.NUM_BEAMS,
                temperature=self.TEMPERATURE,
                max_new_tokens=self.MAX_NEW_TOKENS,
                use_cache=self.USE_CACHE,
                do_sample=self.DO_SAMPLE,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for idx, output_text in zip(jn_idxs, output_texts):
                jersey_number = self.extract_numbers(output_text)
                jersey_number_detection[idx]  = jersey_number
                jersey_number_confidence[idx] = 1.0 if jersey_number is not None else 0.0
                if self.save_jersey_number_full_detection:
                    jersey_number_full_detection[idx] = output_text

        # ════════════════════════════════════════════════════════════════
        # Step 2 – Role detection
        # Option A (skip_role_inference=True):  re-use PRTReId role output.
        # Option B (skip_role_inference=False): run Qwen on stride-selected.
        # ════════════════════════════════════════════════════════════════
        if self.skip_role_inference:
            # Propagate PRTReId role that was already written to detections
            if 'role_detection' in detections.columns:
                role_detection = detections['role_detection'].tolist()
            if 'role_confidence' in detections.columns:
                role_confidence = detections['role_confidence'].tolist()
            # (If columns absent, keep None/0.0 defaults from above)
        else:
            # Run Qwen role inference on stride-selected detections only
            role_idxs = [i for i, sel in enumerate(stride_mask) if sel]

            if len(role_idxs) > 0:
                images = [batch['images'][idx].cpu().numpy() for idx in role_idxs]
                crops  = [batch['crops'][idx].cpu().numpy()  for idx in role_idxs]

                messages = [
                    [{"role": "user", "content": [
                        {"type": "image", "image": Image.fromarray(image)},
                        {"type": "image", "image": Image.fromarray(crop)},
                        {"type": "text",  "text":  self.role_text_prompt},
                    ]}]
                    for image, crop in zip(images, crops)
                ]
                texts = [
                    self.processor.apply_chat_template(msg, tokenize=False,
                                                       add_generation_prompt=True)
                    for msg in messages
                ]
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=texts, images=image_inputs, videos=video_inputs,
                    padding=True, return_tensors="pt",
                ).to(self.device)

                generated_ids = self.model.generate(
                    **inputs,
                    num_beams=self.NUM_BEAMS,
                    temperature=self.TEMPERATURE,
                    max_new_tokens=self.MAX_NEW_TOKENS,
                    use_cache=self.USE_CACHE,
                    do_sample=self.DO_SAMPLE,
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_texts = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                for idx, output_text in zip(role_idxs, output_texts):
                    role = self.extract_role(output_text)
                    role_detection[idx]  = role
                    role_confidence[idx] = 1.0 if role is not None else 0.0

        # ── Write results back to detections DataFrame ───────────────────
        detections['jersey_number_detection']  = jersey_number_detection
        detections['jersey_number_confidence'] = jersey_number_confidence
        if self.save_jersey_number_full_detection:
            detections['jersey_number_full_detection'] = jersey_number_full_detection
        detections['role_detection']  = role_detection
        detections['role_confidence'] = role_confidence

        return detections
