Question: {question}
Predicted Answer: {answer}
Ground Truth Answer: {gt}
  
Please evaluate if the predicted answer is correct compared to the ground truth.
Score the answer on:
Binary correctness (0-1): 1 if the answer is correct, 0 if it is incorrect

Return only a string with these scores in a dictionary and can be parsed by json.loads, e.g. {{"binary_correctness": 1}}
