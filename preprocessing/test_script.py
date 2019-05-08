from preprocessing_open_smile import AudioDataSet

complete_dataset = AudioDataSet(load_training_data=True)
print(complete_dataset.dataset[0:10])
print('annotations length:', len(complete_dataset.annotations))
print('Dataset length:', len(complete_dataset.dataset))