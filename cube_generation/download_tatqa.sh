mkdir dataset
mkdir dataset/tatqa
wget https://raw.githubusercontent.com/NExTplusplus/tat-qa/master/dataset_raw/tatqa_dataset_dev.json -O tatqa_dataset_dev.json
wget https://raw.githubusercontent.com/NExTplusplus/tat-qa/master/dataset_raw/tatqa_dataset_train.json -O tatqa_dataset_train.json
mv tatqa_dataset_train.json dataset/tatqa/
mv tatqa_dataset_dev.json dataset/tatqa/