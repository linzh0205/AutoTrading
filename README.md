# AutoTrading
# DSAI HW2

## Method

## Data analysis

## Model training

## Strategy

```
   Holding stock  | Open price| Predict price | Action
 |----------------|-----------|---------------|-------|
 |       1        |    低     |    高         |    0  |
 |       1        |    高     |    低         |   -1  |
 |       0        |    低     |    高         |    1  |
 |       0        |    高     |    低         |   -1  |
 |      -1        |    低     |    高         |    1  |
 |      -1        |    高     |    低         |    0  |
 |----------------------------------------------------|
```

## Run the code
環境
Python 3.7.1
```
conda create -n test python==3.7
```
```
activate test
```
路徑移至requirements.txt所在的資料夾，輸入安裝套件指令:
```
conda install --yes --file requirements.txt
```
將trader.py、training.csv、testing testing.csv、output.csv載下後(需在同資料夾內)

輸入以下指令:
```
python trader.py --training training.csv --testing testing.csv --output output.csv
```
