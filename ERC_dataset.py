from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence
import random
import json
from collections import deque

    
class Emory_loader(Dataset):
    def __init__(self, txt_file, dataclass, data_type):
        self.dialogs = []
        
        f = open(txt_file, 'r')
        dataset = f.readlines()
        f.close()
        """sentiment"""
        # 'Joyful', 'Mad', 'Neutral', 'Peaceful', 'Powerful', 'Sad', 'Scared'
        pos = ['Joyful', 'Peaceful', 'Powerful']
        neg = ['Mad', 'Sad', 'Scared']
        neu = ['Neutral']
        emodict = {'Joyful': "joy", 'Mad': "mad", 'Peaceful': "peaceful", 'Powerful': "powerful", 'Neutral': "neutral", 'Sad': "sad", 'Scared': 'scared'}
        self.sentidict = {'positive': pos, 'negative': neg, 'neutral': neu}
        context = []
        context_speaker = []
        self.speakerNum = []
        self.emoList = sorted(["joy", "mad", "peaceful", "powerful", "neutral", "sad", 'scared'])
        self.sentList = sorted(["positive", "negative", "neutral"])
        
        self.classCounts = [0 for _ in range(len(self.emoList))]
        self.sentCounts = [0 for _ in range(len(self.sentList))]
        window_size = 12
        try:
            with open('./speaker_list.txt', 'r') as file:
                # Read the content of the file
                speaker_content = file.read()
                # Split the content into individual names
                speaker_list = speaker_content.splitlines()

        except FileNotFoundError:
            # If the file does not exist, continue without any error
            print("there is no such file")
            speaker_list = []        
        
        if data_type != 'train':
            #use uttr histroy only if it is dev/test. For each epoc in training, build fresh history.
            try:  
                with open('./uttr_history.json', 'r') as file:
                    uttr_history = json.load(file)
                    # Convert deque values back to deque objects
                    for speaker, uttr in uttr_history.items():
                        uttr_history[speaker] = deque(uttr, maxlen=window_size)
            except FileNotFoundError:
                uttr_history = {}
        else:
            uttr_history = {}
            
        for i, data in enumerate(dataset):
            if data == '\n' and len(self.dialogs) > 0:
                self.speakerNum.append(len(context_speaker))
                context = []
                context_speaker = []
                continue
            speaker, uttr, emo = data.strip().split('\t')
            context.append(uttr)
            
            if emo in pos:
                senti = "positive"
            elif emo in neg:
                senti = "negative"
            elif emo in neu:
                senti = "neutral"
            else:
                print('ERROR emotion&sentiment')
            
            if speaker not in speaker_list:
                speaker_list.append(speaker)
            speakerCLS = speaker_list.index(speaker)
            context_speaker.append(speakerCLS)
            
            if speaker in uttr_history:
                # If the key already exists, append the value to the existing deque
                uttr_history[speaker].append(uttr)
            else:
                # If the key doesn't exist, create a new deque with maxlen=5
                uttr_history[speaker] = deque(maxlen=window_size)
                uttr_history[speaker].append(uttr)
            
            speaker_utt_history = list(uttr_history[speaker])
            self.dialogs.append([context_speaker[:], context[:], speaker_utt_history[:], emodict[emo], senti])
            self.classCounts[self.emoList.index(emodict[emo])]+=1
            self.sentCounts[self.sentList.index(senti)]+=1
             
        if dataclass == 'emotion':
            self.labelList = self.emoList
            self.class_weights = [min(self.classCounts) / count for count in self.classCounts]
        else:
            self.labelList = self.sentiList        
            self.class_weights = [min(self.sentCounts) / count for count in self.sentCounts]

        self.speakerNum.append(len(context_speaker))
        with open("./speaker_list.txt", 'w') as file:
            for speaker in speaker_list:
                file.write(speaker + '\n')
        
        if data_type == 'train':
            # save only if it is a train, do not want to save the uttr history of dev/test
            with open("./uttr_history.json", 'w') as file:
                uttr_history_dict = {}
                for key, value in uttr_history.items():
                    uttr_history_dict[key] = list(value)
                json.dump(uttr_history_dict, file)

    def get_class_weights(self):
        return self.class_weights  
        
    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx], self.labelList, self.sentidict
    
    
