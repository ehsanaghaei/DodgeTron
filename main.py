from paths import PATH_data, PATH_infostealer
from read_data import list_files, read_files, build_representation, rep_summarizer

# ------------------	 Read Data 	------------------
step_ = "Read File"
print(f"[{step_}] -> Read malware json files")
families, fnames = list_files(PATH_infostealer)
malwares = read_files(fnames[100:110] + fnames[1000:1010] + fnames[2000:2010])

print(f"[{step_}] -> Convert to sequence representation")
representations = build_representation(malwares, segments=['api'])

print(f"[{step_}] -> Summarize the representation")
representations_sum = rep_summarizer(representations)

