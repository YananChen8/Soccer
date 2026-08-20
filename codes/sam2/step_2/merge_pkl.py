import os
import pickle
import json
import argparse
from utils import (
    generate_refined_pklz,
)

# Environment configuration
# Fall back to CPU for unsupported operations when using Apple MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    parser = argparse.ArgumentParser(description='Merge individual video results into aggregated pkl/pklz files')
    parser.add_argument('--input_pklz', type=str, required=True,
                        help='Input pklz file path containing tracklets')
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='Directory containing video frames')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--split', type=str, default='test',
                        help='Data split (train, valid, test)')
    parser.add_argument('--video_id_list', type=str,
                        help='Comma-separated list of specific video IDs to process, e.g., "116,117,118"')
    parser.add_argument('--video_id_start', type=int,
                        help='Starting video ID for range processing (inclusive)')
    parser.add_argument('--video_id_end', type=int,
                        help='Ending video ID for range processing (inclusive)')
    parser.add_argument('--exempt_video_id_list', type=str,
                        help='Comma-separated list of video IDs to exempt from processing, e.g., "120,121,122"')
    parser.add_argument('--save_results', action='store_true',
                        help='Save aggregated results to a pkl file')
    parser.add_argument('--output_pkl', type=str, default=None,
                        help='Path to save the output pkl file (default: <output_dir>/results.pkl)')
    parser.add_argument('--save_refined_pklz', action='store_true',
                        help='Save a refined version of the input pklz file with matched track IDs')
    parser.add_argument('--save_pklz_path', type=str, default=None,
                        help='Path where the refined pklz file will be saved (default: <output_dir>/refined_<input_pklz_basename>)')
    parser.add_argument('--include_unmatched_segments', action='store_true',
                        help='Include unmatched segments as new rows in the refined pklz file')
    parser.add_argument('--fix_duplicate_track_ids', action='store_true',
                        help='Check and fix duplicate track_ids in each frame by adding 100')
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load metadata to determine the list of video IDs to process
    video_ids = []
    if args.video_id_list:
        # If specific video IDs are provided, only process those
        video_ids = [vid.strip() for vid in args.video_id_list.split(',')]
        print(video_ids)
    elif args.video_id_start is not None and args.video_id_end is not None:
        # If a video ID range is specified, generate all video IDs within the range
        if args.video_id_start > args.video_id_end:
            raise ValueError(f"video_id_start ({args.video_id_start}) must not be greater than video_id_end ({args.video_id_end})")
        video_ids = [str(vid) for vid in range(args.video_id_start, args.video_id_end + 1)]
        print(f"Generated video ID list from range: {args.video_id_start}-{args.video_id_end}, total {len(video_ids)} videos")
    elif args.video_id_start is not None or args.video_id_end is not None:
        # If only one range parameter is specified, raise an error
        raise ValueError("Both --video_id_start and --video_id_end must be specified together")
    elif args.dataset_root:
        # Otherwise, load all video IDs from metadata
        try:
            with open(os.path.join(args.dataset_root, "sequences_info.json"), 'r') as f:
                metadata = json.load(f)
                
            # Get video IDs for the specified split
            if args.split == 'valid':
                split_key = 'validation'
            else:
                split_key = args.split
                
            video_ids = [vid["name"].split('-')[1] for vid in metadata[split_key]]
            print(f"Found {len(video_ids)} videos from metadata")
        except Exception as e:
            print(f"Failed to load metadata: {e}")
            if args.video_id_list is None:
                raise ValueError("Must provide --video_id_list or a valid --metadata_path")
            
    # Create directory for storing individual video results
    video_results_dir = os.path.join(args.output_dir, "video_results")
    os.makedirs(video_results_dir, exist_ok=True)

    # Exempt list: these videos will not be included in the output results
    exempt_video_ids = []
    if args.exempt_video_id_list:
        exempt_video_ids = [vid.strip() for vid in args.exempt_video_id_list.split(',')]
        print(f"The following videos will be exempted from processing: {exempt_video_ids}")

    # Save results to pkl file
    if args.save_results:
        output_pkl_path = args.output_pkl if args.output_pkl else os.path.join(args.output_dir, "results.pkl")
        print(f"Aggregating all video results to: {output_pkl_path}")
        
        # Aggregate results from all videos, excluding exempt ones
        all_results = {}
        for video_id in video_ids:
            # Skip exempt videos
            if video_id in exempt_video_ids:
                print(f"Video {video_id} is in the exempt list, excluded from aggregated results")
                continue
                
            video_result_path = os.path.join(video_results_dir, f"{video_id}_result.pkl")
            print(video_id)
            with open(video_result_path, 'rb') as f:
                all_results[video_id] = pickle.load(f)
        
        # Save aggregated results
        with open(output_pkl_path, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"Aggregated results saved to: {output_pkl_path}")
    
    # Save refined pklz file
    if args.save_refined_pklz:
        # Load results from all videos, excluding exempt ones
        all_results = {}
        for video_id in video_ids:
            # Skip exempt videos
            if video_id in exempt_video_ids:
                print(f"Video {video_id} is in the exempt list, will not be processed, copying directly from input pklz")
                continue
                
            video_result_path = os.path.join(video_results_dir, f"{video_id}_result.pkl")
            with open(video_result_path, 'rb') as f:
                all_results[video_id] = pickle.load(f)
                
        refined_pklz_path = generate_refined_pklz(args, all_results, exempt_video_ids)
        print(f"Refined pklz file saved to: {refined_pklz_path}")
    
    print("All videos processed successfully")



# Run main function
if __name__ == "__main__":
    main()
