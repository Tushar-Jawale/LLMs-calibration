#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import json
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial


# # Load the dataset

# In[4]:


df = pd.read_csv('dataset/simpleQA.csv')


# In[5]:


df.head()


# In[6]:


df.tail(10)


# # Prompt for generating distractors

# In[7]:


prompt = """You are an expert synthetic data generator. Your task is to generate three plausible but incorrect answers to a given question.

Guidelines for generating wrong answers:
1. Each answer should be factually incorrect but plausible within the context
2. Match the answer type (e.g. if asking for a date, provide wrong dates)
3. The wrong answers should be clearly distinct from the correct answer and from each other
4. Maintain a similar level of specificity as the original answer
5. The answers should be realistic and not obviously wrong

Example 1:
Question: What is the capital of France?
Answer: Paris
Wrong Answers: 
- Lyon
- Marseille 
- Bordeaux
Reason: All are major French cities, but incorrect as capital

Example 2:
Question: Who was the first president of the United States?
Answer: George Washington
Wrong Answers:
- John Adams
- Thomas Jefferson
- Benjamin Franklin
Reason: All are founding fathers but not the first president

Example 3:
Question: In what year did World War II end?
Answer: 1945
Wrong Answers:
- 1943
- 1944
- 1946
Reason: All are plausible years during or near WWII but not when it ended

Example 4:
Question: Who wrote Romeo and Juliet?
Answer: William Shakespeare
Wrong Answers:
- Christopher Marlowe
- Ben Jonson
- John Webster
Reason: All are prominent Elizabethan playwrights

Example 5:
Question: What is the largest planet in our solar system?
Answer: Jupiter
Wrong Answers:
- Saturn
- Neptune
- Uranus
Reason: All are gas giant planets, but smaller than Jupiter

Please generate three wrong answers that follow these guidelines for the given question.
The answers should be:
- Factually incorrect but plausible
- Match the same answer type (e.g. date, person, number)
- Clearly distinct from the correct answer and each other
- Similar in specificity/detail level
- Realistic and not obviously wrong

Return only three wrong answers as a list in JSON format with the following requirements:
- Each wrong answer should be a string
- The output should be a single JSON object with key "wrong_answers" 
- The value should be an array of exactly 3 wrong answers
- No explanations or additional text should be included
- The answers should maintain consistent formatting with the correct answer

Example format:
{{
    "wrong_answers": ["Wrong Answer 1", "Wrong Answer 2", "Wrong Answer 3"]
}}

Question: {question}
Correct Answer: {answer}
Generate three wrong answers:
"""


# # LLM call to generate distractors

# In[ ]:


from openai import OpenAI
client = OpenAI(api_key="---")


# In[ ]:


def generate_wrong_answer(question, answer):
    """
    Generate 3 plausible but incorrect answers for a given question using GPT-4.

    Args:
        question (str): The question to generate wrong answers for
        answer (str): The correct answer to the question

    Returns:
        list: List of 3 wrong answers, or empty list if generation fails

    The function will retry up to 3 times if the API call fails.
    Wrong answers are generated to be:
    - Factually incorrect but plausible
    - Match the same answer type as correct answer
    - Clearly distinct from correct answer and each other
    - Similar in specificity/detail level
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            updated_prompt = prompt.format(question=question, answer=answer)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": updated_prompt}],
                temperature=1,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)['wrong_answers']
        except Exception as e:
            print("Error: ", e)
            if attempt == max_retries - 1:
                raise e
            continue

    return []

print(generate_wrong_answer("What is the capital of India?", "New Delhi"))


# In[ ]:


index_incorrect_answers = defaultdict(list)

def process_row(index, df):
    problem = df['problem'][index]
    answer = df['answer'][index]
    wrong_answer = generate_wrong_answer(problem, answer)
    return index, wrong_answer

with ThreadPoolExecutor(max_workers=8) as executor:
    for index, wrong_answer in tqdm(
        executor.map(partial(process_row, df=df), range(len(df))), 
        total=len(df)
    ):
        index_incorrect_answers[index] = wrong_answer



# In[ ]:


index_incorrect_answers


# In[ ]:


with open('index_incorrect_answers_final.json', 'r') as f:
    index_incorrect_answers_final = json.load(f)

len(index_incorrect_answers_final)


# # Create a new dataframe with the wrong answers

# In[ ]:


df_wrong = df.copy()
df_wrong['wrong_answer_1'] = df_wrong.index.map(lambda x: index_incorrect_answers_final[str(x)][0])
df_wrong['wrong_answer_2'] = df_wrong.index.map(lambda x: index_incorrect_answers_final[str(x)][1]) 
df_wrong['wrong_answer_3'] = df_wrong.index.map(lambda x: index_incorrect_answers_final[str(x)][2])

df_wrong.head(20)


# In[ ]:


df_wrong.to_csv('synthetic_dataset_with_wrong_answers.csv', index=False)

