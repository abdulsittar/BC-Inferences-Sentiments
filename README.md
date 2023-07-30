# Profiling the news spreading barriers using news headlines

## Abstract
>News headlines can be a good data source for detecting the barriers to the spreading of news in news media, which be useful in many real-world applications. In this paper, we utilize semantic knowledge through the inference-based model COMET and the sentiments of news headlines for barrier classification. We consider five barriers, including cultural, economic, political, linguistic, and geographical, and different types of news headlines, including health, sports, science, recreation, games, homes, society, shopping, computers, and business. To that end, we collect and label the news headlines automatically for the barriers using the metadata of news publishers. Then, we utilize the extracted commonsense inferences and sentiments as features to detect the barriers to the spreading of news. We compare our approach to the classical text classification methods, deep learning, and transformer-based methods. The results show that 
1) The inference-based semantic knowledge provides distinguishable inferences across the ten categories that can increase the effectiveness and enhance the speed of the classification model.
2), The news of positive sentiments crosses the political barrier whereas the news of negative sentiments crosses the cultural, economic, linguistic, and geographical barriers,
3) the proposed approach using inferences-based semantic knowledge and sentiment offers better performance than the usual (the average F1-score for 4 out of 5 barriers has significantly improved. For cultural barriers from 0.41 to 0.47, for economic barriers from 0.39 to 0.55, for political barriers from 0.59 to 0.70 and for geographical barriers from 0.59 to 0.76.) for classifying the barriers to the spreading of news.

## Requirements

We recommend Conda with Python3. Use requirements.txt to create the necessary environment. 

