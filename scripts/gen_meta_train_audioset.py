import pandas as pd
import subprocess
import os
df = pd.read_csv("./data/AudioSet/audioset_train_strong_with_label.tsv",sep = "\t")
print(df)

# Filtered Sounds

# tomofun -> audioset name -> audioset tag
# Barking -> Bark -> /m/05tny_
# Howling -> Howl -> /m/07qf0zm
# Crying -> Whimper (dog) -> /t/dd00136
# CO_Smoke -> Smoke detector, smoke alarm -> /m/01y3hg
# GlassBreaking -> Glass Shatter -> /m/07rn7sz
# Doorbell -> Doorbell -> /m/03wwcy
# Bird -> Bird -> /m/015p6
# Music_Instrument -> X
# Laugh_Shout_Scream -> Laughter, Bellylaugh, Shout, Screaming -> /m/01j3sz, /m/07sq110, /m/07p6fty, /m/03qc9zr
# Others -> X
#- Vacuum -> Vacuum cleaner -> /m/0d31p
#- Blender -> Blender, food processor -> /m/02pjr4
#- Electrics -> Lawn mower, Electric shaver -> /m/01yg9g, /m/02g901
#- Cat -> Cat /m/01yrx
#- Dishes -> Dishes, pots, and pans -> /m/04brg2

# Background noise -> /m/093_4n
audioset2tomofun = {
	"/m/05tny_":"0Barking",
	"/m/07qf0zm":"1Howling",
	"/t/dd00136":"2Crying",
	"/m/01y3hg":"3CO_Smoke",
	"/m/07rn7sz":"4GlassBreaking",
	"/m/03wwcy":"6Doorbell",
	"/m/015p6":"7Bird",
	"/m/01j3sz":"9Laugh_Shout_Scream",
	"/m/07sq110":"9Laugh_Shout_Scream",
	"/m/07p6fty":"9Laugh_Shout_Scream",
	"/m/03qc9zr":"9Laugh_Shout_Scream",
	"/m/0d31p":"5Vacuum",
	"/m/02pjr4":"5Blender",
	"/m/01yg9g":"5Electrics",
	"/m/02g901":"5Electrics",
	"/m/01yrx":"5Cat",
	"/m/04brg2":"5Dishes",
}
same_video_count = {
	
}
Filename = []
Label = []
Remark = []
for row in df.itertuples():
	if row.label in audioset2tomofun:
		start = row.start_time_seconds
		end = row.end_time_seconds
	else:
		continue
	print(row)
	 
	duration = end - start
	if duration > 5:
		mid = (start + end) / 2
		start = mid - 2.5
		duration = 5
	page = "_".join(row.segment_id.split("_")[:-1])
	if page not in same_video_count:
		same_video_count[page] = 1
	else:
		same_video_count[page] += 1
	filename = f"./data/AudioSet/raw/{page}_{same_video_count[page]}.wav"
	if os.path.exists(filename):
		print(filename)
	else:
		continue
	Filename.append(f"{page}_{same_video_count[page]}")
	Label.append(audioset2tomofun[row.label][0])
	Remark.append(audioset2tomofun[row.label][1:])
print(Label)
dfout = pd.DataFrame()
dfout["Filename"] = Filename
dfout["Label"] = Label
dfout["Remark"] = Remark
dfout.to_csv("./data/AudioSet/meta_train.csv",index = False)