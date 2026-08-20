
import inspect, trackeval
cls = trackeval.datasets.SoccerNetGS
print("FILE:", inspect.getfile(cls))
dc = cls.get_default_dataset_config()
for k in ["EVAL_SPACE","EVAL_DIST_TOL","USE_ROLES","USE_TEAMS","USE_JERSEY_NUMBERS","THRESHOLD"]:
    print("DEFAULT", k, "=", dc.get(k))
# print the similarity method source
for name in ["_calculate_similarities","get_pitch_distance","_get_similarity","calculate_similarities"]:
    m=getattr(cls,name,None)
    if m:
        print("==== "+name+" ====")
        try: print(inspect.getsource(m))
        except Exception as e: print("src err",e)
