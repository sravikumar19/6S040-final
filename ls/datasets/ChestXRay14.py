import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from ls.utils.print import print


class ChestXRay14(Dataset):
    def __init__(self, task: str = 'DIAGNOSIS'):
        '''
            We use the ChestXRay Machine Learning Data Set in Google Cloud Storage. 


            root: path to download/load the data.
            task: a specific diagnosis prediction task that we want to load.
                Options: ["Atelectasis", "Consolidation", "Infiltration",
                       "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
                       "Effusion", "Pneumonia", "Pleural_Thickening",
                       "Cardiomegaly", "Nodule", "Mass", "Hernia", "No Finding", "All"]
        '''

        assert task in ["Atelectasis", "Consolidation", "Infiltration",
                        "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
                        "Effusion", "Pneumonia", "Pleural_Thickening",
                        "Cardiomegaly", "Nodule", "Mass", "Hernia", "No Finding", "All"], \
            f"Diagnosis {task} is not supported in ChestXRay14."

        self.csv = pd.read_csv(
            "/content/gdrive/MyDrive/chestxray14-data/sample_labels.csv")
        self.image_paths, self.targets = ChestXRay14.load_data(self, task)
        self.length = len(self.targets)
        self.preprocessing = ChestXRay14.get_transform_cub()

    @staticmethod
    def get_transform_cub():
        '''
            Transform the raw images so that it matches with the distribution of
            ImageNet
        '''

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        return transform

    @staticmethod
    def load_data(self, task: str):
        '''
            Load the ChestXray14 dataset files.
        '''

        pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                       "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
                       "Effusion", "Pneumonia", "Pleural_Thickening",
                       "Cardiomegaly", "Nodule", "Mass", "Hernia"]

        self.csv = self.csv.groupby("Patient ID").first()
        self.csv = self.csv.reset_index()

        # getting patient id, age, and sex
        self.csv["patientid"] = self.csv["Patient ID"].astype(str)
        # self.csv['age_years'] = self.csv['Patient Age'][:-1] * 1.0
        self.csv['sex_male'] = self.csv['Patient Gender'] == 'M'
        self.csv['sex_female'] = self.csv['Patient Gender'] == 'F'

        df = self.csv
        image_paths = list(df['Image Index'])
        targets = []
        if task != "All":
            labels = list(df['Finding Labels'])
            for i in range(len(labels)):
                if task in labels[i]:
                    targets.append(1)
                else:
                    targets.append(0)
        else:
            for pathology in pathologies:
                targets.append(
                    df["Finding Labels"].str.contains(pathology).values)

            targets = np.asarray(targets).T
            targets = targets.astype(np.float32)

        targets = torch.tensor(targets)
        return image_paths, targets

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        '''
            Given the index, return the x-ray and the binary label.
        '''

        img = Image.open('/content/gdrive/MyDrive/chestxray14-data/images/' +
                         self.image_paths[idx]).convert('RGB')
        img = self.preprocessing(img)
        target = self.targets[idx]
        return img, target
