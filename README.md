# AutoTrading
# DSAI HW2

## Data analysis
   擷取出close、open、high、low的資料並將其可視化。
   
   training data 中close、open、high、low趨勢圖:
   
   ![4line](https://github.com/linzh0205/AutoTrading/blob/main/plot/4line.jpeg)
   
   
   testing data 中 close、open趨勢圖:
   
   ![close_open](https://github.com/linzh0205/AutoTrading/blob/main/plot/Figure_1.png)
   
   由上圖可以發現close、open之間存在延遲一天的關係故兩者具有一定的關係存在，因此將使用close、open作為模型訓練的特徵。
   
## Method & Model training
   由上述資料分析後選擇close、open作為特徵，並以常用於預測時間序列資料的模型LSTM作為此次的訓練模型。
   
   需要預測20天的股票開盤價，因此在訓練資料中以每20筆資料預測1筆的方式去做模型的訓練。
   
   LSTM Model Summary:
   
   ![LSTM](https://github.com/linzh0205/AutoTrading/blob/main/plot/LSTM.JPG)
   
   Predict Result:
   
   ![result](https://github.com/linzh0205/AutoTrading/blob/main/plot/)
   
## Trader Strategy
   判斷目前股票數量為0、1或-1，並以買入之收盤價與預測收盤價比較，以此為依據判斷動作為買、賣、持有
   
   並記錄每一次買入時所花費的價錢。
   
   從training data中可以發現，每天漲跌起伏大多都在1塊左右，因此在20天的action中，將買入股票的開盤價與預測的開盤價比價差
   
   當價差大於0.6塊時賣出，反之持有。
   
   ex1:持有股票數量為0，預測開盤價<實際開盤價(跌)則買入，action為1，反之賣空，action為-1。
   
   ex2:持有股票數量為-1，預測開盤價與賣空價價差小於0.6則買入，action為1，反之持有，action為0。
   
   ex3:持有股票數量為1，當時買入此張股票的開盤價與預測開盤價價差小於0.6則繼續持有，action為0，反之賣出，action為-1。
   
```
   | 持有股票 | 預測開盤價與買進開盤價比較 | 動作 |
   --------------------------------------------
   |    0    |    預測開盤價<實際開盤價  |   1  |
   |------------------------------------------|
   |    0    |    預測開盤價>實際開盤價  |  -1  |
   |------------------------------------------|
   |    1    |    預測開盤價<買進開盤價  |   0  |
   |------------------------------------------|
   |    1    |    預測開盤價>買進開盤價  |  -1  |
   |------------------------------------------|
   |   -1    |    預測開盤價<賣空價      |   1  |
   |------------------------------------------|
   |   -1    |    預測開盤價>賣空價      |   0  | 
   |------------------------------------------|
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
