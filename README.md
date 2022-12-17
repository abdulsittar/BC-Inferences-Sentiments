# Profiling the news spreading barriers using news headlines

## Abstract
>News headlines can be a good data source for detecting the news spreading barriers in news media, which may be useful in many real-world applications. In this paper, we utilize semantic knowledge through the inference-based model COMET and sentiments of news headlines for barrier classification. We consider five barriers including cultural, economic, political, linguistic, and geographical, and different types of news headlines including health, sports, science, recreation, games, homes, society, shopping, computers, and business. To that end, we collect and label the news headlines automatically for the barriers using the metadata of news publishers. Then, we utilize the extracted commonsense inferences and sentiments as features to detect the news spreading barriers. We compare our approach to the classical text classification methods, deep learning, and transformer-based methods. The results show that the proposed approach using inferences-based semantic knowledge and sentiment offers better performance than the usual (the average F1-score of the ten categories improves from 0.41, 0.39, 0.59, and 0.59 to 0.47, 0.55, 0.70, and 0.76 for the cultural, economic, political, and geographical respectively) for classifying the news-spreading barriers

## Requirements

We recommend Conda with Python3. Use requirements.txt to create the necessary environment. 

