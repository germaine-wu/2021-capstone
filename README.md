# Machine Learning based Pattern Recognition in Mission-critical Systems Scheduling
2021-S1-COMP5703  CS4-2
## Project description:
This project aims to discover meaningful patterns between `mission-critical` systems and their `optimal scheduling strategies` using `machine learning`. 
A mission-critical system can be represented as a vector of numeric parameters, and its corresponding optimal scheduling strategy is a well-designed sequence of instructions. 
There is a `binding` link between mission-critical systems and their optimal scheduling strategies, but the relation is unknown in advance.   
In this project, our group will be given `a dataset containing rows`, and each row contains a pair. In each pair, it has `‘features’` and `‘the target sequence’`. 
We need to train `multiple machine learning models` to predict the targets given features, report all training processes, and the `performance` of the fine-tuned trained models.  
`Expected outcome:`  
• Source code for training the fine-tuned machine learning models  
• A report containing all training processes, and the performance of the fine-tuned trained models  
<br/> 
## Project Files  
• `/model`: store pre-trained models of two model methods  
&#8195;/model1 is for `model1` models;  
&#8195;/model2 is for `model2` models  
• `/Training Data`: training sample data set for pre-trained models  
&#8195;(already have pre-trained models, if want to train models with new training data, replace data in this folder)  
• `/Test Data`: test data  
&#8195;(two default test data existed, one for normal sequence; another for long sequence)  
• `main.py`: main part of our code project  
• `model1.py`: training process for model 1 method  
• `model2.py`: training process for model 2 method  
• `prediction.py`: prediction structure & models combination   
• `validation.py`: validation function to get the results  
• `config.py`: setting of project parameters  
<br/> 
## Code implmentation
Command line `python main.py` to run our program.  
<br/> 
Function list:  
• `--function` / `-f`: &#8194; values: `Pred` or `Eval`  
&#8195;Choose the predicte or evaluate function  
• `--input` &#8195;&#160;/ `-i`: &#8194; values: `Input data file name`   
&#8195;Input the data to our model  
• `--output` &#8194;&#160;/ `-o`: &#8194; values: `Output data file name`   
&#8195;Define the output file name of predicted output  
• `--trueout` &#160;/ `-t`: &#8194; values: `True output file name`    
&#8195;Input the true output file name  (for evaluation)  
• `--result` &#8194;&#160;/ `-r`: &#8194; values: `Result file name`   
&#8195;Define the evaluation result file name  (for evaluation)  
• `--method` &#8194;&#160;/ `-m`: &#8194; values: `1` or `2`   
&#8195;Choose the models & method : 1 as model1; 2 as model2  (for prediction)  
<br/> 
If use `Pred` function, need to input attributes: `function`, `input`, `output`, `method`  
For example, `python main.py -f Pred -i test_long_input.txt -o output.txt -m 2` to predict the output sequences `output.txt`   
<br/>
If use `Eval` function, need to input attributes: `function`, `input`, `output`, `trueout`, `result`  
For example, `python main.py -f Eval -i test_long_input.txt -o output.txt -t test_long_output.txt -r result.txt` to get the result `result.txt`   


