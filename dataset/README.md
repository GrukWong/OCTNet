# Dataset Setup

Download **Retinal OCT C8** from Kaggle and place the extracted dataset here:

```text
data/RetinalOCT_Dataset/
├── train/
│   ├── AMD/
│   ├── CNV/
│   └── ...
├── val/
│   ├── AMD/
│   ├── CNV/
│   └── ...
└── test/
    ├── AMD/
    ├── CNV/
    └── ...
```

Expected class names, in ImageFolder alphabetical order: `AMD`, `CNV`, `CSR`,
`DME`, `DR`, `DRUSEN`, `MH`, and `NORMAL`.

The dataset is not included in this repository. Please review the dataset page
and its license before downloading or redistributing it.
