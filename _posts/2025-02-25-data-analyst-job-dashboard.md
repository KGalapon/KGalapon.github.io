---
title: "Data Analyst Job Statistics Dashboard: An End-to-End Project"
date:  2025-02-25
categories: ['Project']
tags: ['Python','Pandas', 'Streamlit', 'Plotly', 'Scikit-Learn', 'Web Scraping', 'Data Warehousing', 'LLMs', 'NLP','Machine Learning']
math: true
---

<iframe width="100%" height="400" 
    src="https://www.youtube.com/embed/tP5oP3EMXc0" 
    frameborder="0" 
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
    allowfullscreen>
</iframe>


I programmed a web scraper, stored the data in a cloud database, clustered job description through machine learning, and created a dashboard web application to summarize my findings. You can visit the app [here](https://data-analyst-jobs-dashboard.streamlit.app/). If the app is currently down, do not hesitate to get it back up or contact me: **karlchestergalapon77@gmail.com**. 

As an aspiring data scientist, I had been tirelessly asking myself countless questions like:

- What are the most important skills/technologies to learn for data analysts in the Philippines?
- What does the distribution of salary and experience requirements across data analytics job offerings look like?
- Are there different kinds of data analyst/science roles across industries?

I scoured the internet for answers, but none of the available ones had the detail I sought. Ideally, my goal was a tool that would allow me to **simultaneously see the Philippine data analytics job market.**


## Data Extraction, Preprocessing, and Storage

### Scraping

I needed a lot of data. I could not resort to online datasets because I required insights to be as recent as possible. Hence, I learned web scraping, particularly the **BeautifulSoup** library for Python.

Constructing the scraping algorithm was simple but time-consuming. A specific script had to be written to accommodate unique website architectures. I had even foregone scraping some sites because they were [dynamic](https://en.wikipedia.org/wiki/Dynamic_web_page). Even so, I successfully gathered sufficient data for analysis. You can see the script for my scraper in my [GitHub repository](https://github.com/KGalapon/Web-to-BigQuery-JobPost-Scraper/blob/main/job_scraper.py). For each job posting, the scraper searches for the content of specific html tags and collects them:

```python
#COMPANY_NAME
company = soup_job_page.find('span', attrs = {'data-automation': 'advertiser-name'}).text
companies.append(company)   
#LOCATION
location = soup_job_page.find('span', attrs = {'data-automation': 'job-detail-location'}).text
locations.append(location)
```

I focused on the following features: `job_id`, `job_titles`, `company_names`, `locations`, `work_types` (full-time, contractual, part-time), `salaries`, `post_dates`, `job_link`, and the `job_description`.

### Cleaning

After scraping a batch of job listings, I needed to clean them thoroughly. Heavy use of **pandas** and **regular expressions** proved to be sufficient for most data cleaning tasks such as removing duplicates, converting dates into `datetime` objects, removing unncessary components in strings (punctuation, html tags, extra spaces, etc.), and converting text into lowercase. 

However, I wanted information like **years of experience**, **required skills/tools**, and  **desired college courses**, which were embedded in the unstructured job descriptions. It was impossible to program a deterministic set of steps that could obtain these from every text format. I needed a tool that 'understood' text. So, I immediately thought of utilizing LLMs. Using my google account, I obtained an API key from **Google AI Studio** for `gemini-1.5-flash`, which is the free version. Here's an example of one of my prompts to extract data from text:


```python
model = genai.GenerativeModel("gemini-1.5-flash")
prompt = f"""
For the job decription below, place the minimum years of experience as an integer in a python list and nothing else. 
Place 0 if it is open to fresh graduates. If the document does not mentioned years of experience, then the list must be empty. 
The job_description is:
{job_description} """
response = model.generate_content(prompt)
output = response.text
```
All my code for cleaning data and prompts to Gemini can be viewed [here](https://github.com/KGalapon/Web-to-BigQuery-JobPost-Scraper/blob/main/cleaning.py). The functions for cleaning are then applied to every record in the scraped data.

### Storage

After the data has been cleaned, it was written in a **Google Big Query SQL Database** through the `bigquery` method of the `google.cloud` Python library:

```python
job = client.load_table_from_dataframe(dataframe = df, destination=table_id)
```

I compiled all the scripts used in scraping, cleaning, and storing in a single [Python Script](https://github.com/KGalapon/Web-to-BigQuery-JobPost-Scraper/blob/main/scrape_clearn_write.ipynb), which I ran for several days, allowing me to collect over **2000** job listings. The amount was relatively small, but it was sufficient.


## Job Description Topic Clustering

Before performing various clustering methods, I performed standard text preprocessing techniques such as **word segmentation**, **tokenization**, and **stopword removal**  to the job description data using the **nltk** and **wordsegment** packages. Then, I transformed the documents into word embeddings through the `all-MiniLM-L6-v2` sentence transformer of **Hugging Face** and the `Tfidfvectorizer()` of **scikit-learn**.

I also applied **principal component analysis** on both embeddings to obtain a dimension-reduced version of my data.

### Clustering Using KMeans, DBSCAN, Agglomerative Clustering, Birch, and Gaussian Mixture Models

Using **scikit-learn**, I imported all of the unsupervised learning models mentioned above and wrote a customized Grid Search algoritm that looped through several sets of parameters for each model. The best model was selected using the [silhoutte score metric](https://en.wikipedia.org/wiki/Silhouette_(clustering)), which measures how well clusters are separated and how close the data are within clusters.

The notebook containing my script can be found [here](https://github.com/KGalapon/Job_Description_Clustering/blob/main/JobDescriptionClustering.ipynb).

After running GridSearch to both embeddings as well as the PCA-reduced versions, the best clustering came from the `AgglomerativeClustering()` model with **two** clusters, achieving a silhoutte score of $$0.521$$. This score was acceptable given the amount of data I had. Despite this, I ventured toward newer and more sophisticated topic models.

### Topic Modelling Using Latent Dirichlet Allocation (LDA)

Given a fixed number of topics $$K$$, LDA, in as simple a description as possible, learns the how likely topics are in each document and how likely words are from each topic. For a more mathematical/statistical exposition we have the following description from a machine learning YouTube channel, [TwinEd Productions](https://www.youtube.com/watch?v=1_jq_gWFUuQ), written below, which you can skip.

---
Suppose that we have $$M$$ documents and $$K$$ topics; both set by the modeller. LDA is a Bayesian generative topic model that aims to learn the probabilities of topics per document ($$\theta_m$$) and the probabilities of words per topic ($$\varphi_k$$), that is, for all $$m \in \{1,...,M\}$$ and $$k \in \{1,...,K\}$$. As an assumption, $$\theta_m$$ and $$\varphi_k$$ have the following [Dirichlet distributions](https://en.wikipedia.org/wiki/Dirichlet_distribution), $$\text{Dir}\left(\alpha\right)$$ and  $$\text{Dir}\left(\beta\right)$$ respectively, meaning, they are vectors of values from $$0$$ to $$1$$. They are learned through the following scary-looking objective function:

$$\{\hat{\theta_i}\}_{i=1}^{M}, \{\hat{\varphi_i}\}_{j=1}^{K} = \operatorname*{arg\,max}_{\{\theta_i\}_{i=1}^{M}, \{\varphi_i\}_{j=1}^{K} } \prod_{m=1}^{M} \prod_{n=1}^{N} P\left( w_{n,d}\right | \{\varphi_i\}_{j=1}^{K} , \theta_m )$$

$$\text{where } w_{n,d} \text{ is the } n \text{th word of document } m.$$

That is, we want to maximize the probabilities of the words in all documents given the distribution of topics per document and distribution of words per topic. The variables are learned through Monte Carlo simulations like [Gibbs sampling](https://en.wikipedia.org/wiki/Gibbs_sampling).

---


The **gensim** Python library has the `LdaModel()` method, which made clustering easy. But beforehand, I converted the array of documents into a corpus of in **bag-of-words** format and a gensim `Dictionary` object using the **nltk** and **gensim** packages. I looped through various topic numbers to select the ideal amount of clusters:


```python
scores = dict({})
for i in range(2,100):
    lda_model = LdaModel(corpus=corpus, #bag-of-words corpus
                        id2word=id2word, #Dictionary object
                        num_topics=i, #number of topics
                        per_word_topics=True)
    coherence_model_lda = CoherenceModel(model=lda_model, texts=filtered_corpus, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(f'{i} clusters done')
    scores[i] = coherence_lda
```
I used the **coherence score metric**, which measures the similarity of words in a topic ([see this link for more detail](https://neptune.ai/blog/pyldavis-topic-modelling-exploration-tool-that-every-nlp-data-scientist-should-know)) to pick the optimal amount of topics. It turned out that the ideal amount is $$15$$, achieving a coherence score of $$0.4838$$. The notebook containing the script for LDA topic modelling can be found [here](https://github.com/KGalapon/Job_Description_Clustering/blob/main/JobDescriptionLDA.ipynb).

Luckily, Python has an excellent LDA visualization library called **pyLDAvis**, which gave me the following visualization:

![plot](/assets/Media/Data_Analyst_Job_Statistics_Web_App/lda_vis.png)

The blobs in the Latent Dirichlet Allocation  topic visualization represent the projection of topic distributions (vector of probabilities) into two dimensions. Of course, nearer blobs indicate similar topics. The bars on the right represent which words are most important per topic. More specifically, because each topic is assumed as a distribution of words, it shows how much each word was generated or drawn from the given topic.

We see, from the topic clusters above, that **topic two** contains jobs that actively use python ,data analytics, and machine learning, with **data**, **models**, **learning**, and **python** as its salient words.

## The Data Analytics Job Statistics Dashboard

With clustering done and data in the clouid waiting to be analyzed, the final step was to create a dashboard showcasing various summaries of my data and my topic model. The script could be found [here](https://github.com/KGalapon/Job_Analytics_Dashboard/blob/main/Dashboard.py).

This was done without much difficulty using the **Streamlit** framework, which allows for fast and easy development of data web applications. The dashboard and it's various plots can be seen through the [demo at the top of the page](https://www.youtube.com/watch?v=tP5oP3EMXc0).

We observe that the median salary lower bound and upper bound are **45,000** and **55,000** pesos respectively.
 The most common required years of experience across the jobs is **3**.

![plot](/assets/Media/Data_Analyst_Job_Statistics_Web_App/metric.png)

The most desired tools are **SQL**, **Excel**, **Python**, **PowerBI**, **Tableau**, and **AWS**. **Data analysis**, **visualization**, and **modelling** are the most common tasks or responsibilities for data analysts. Additionally, the most sought-after educational backgrounds are **Computer Science** and **Information Technology**. 
  
![plot](/assets/Media/Data_Analyst_Job_Statistics_Web_App/bar_charts.png)
  
The dashboard also corroborates the expected direct relationship between **years of experience** and **salary**. 

![plot](/assets/Media/Data_Analyst_Job_Statistics_Web_App/charts.png)

And these are a just the insights from a first glance at the dashboard.

As shown in the video, the dashboard also allows the user to filter using specific **locations**, **tools or technologies used**, **educational background**, the **topic cluster** of the job description, the **years of experience**, and the **salary**, allowing for more specific information to be obtained.

## Points for Improvement and Future Direction

Of course, the app and the entire project, are far from being a quality product. Despite this, as proof of concept, this project showed that creating even better job data mining projects is indeed possible for me. Here are some steps that I believe will improve the project significantly:

- The lack of data prevented me from training a satisfactory regression or classification model which can predict the salary based on features such as **locations**, **job descriptions**, and **years of experience.** It also inhibited the quality of the topic clusters. To gather more data across various sources, I could try learning more web scraping tools like **Selenium** or **ScraPy**, which work even for dynamic job listings sites. 

- I could also simply scrape data for a longer amount of time, say, several months. I was only able to scrape data for barely two weeks because, even though the scripts for scraping, cleaning, and storing were combined, I still had to manually run and monitor them for errors.  I could try deploying a more error-proofed data preprocessing routine to **GitHub Actions**, allowing the code to run hands-free every day.

- Making the modelling process autonomous through **sequential learning**, wherein the topic clustering model can be updated based on small batches of data, will also help the process become more real-time. However, I have not found nor figured out a way to do that yet.

Neverthelesss, I am quite proud of this project. It was able to deliver the results I wanted and answer the orignal questions I had. I also certainly grew as a data science practitioner along the way. Some important learnings were:

- creating web scraping scripts using **BeautifulSoup** 
- writing data to databases hosted by a cloud service like **Google BigQuery**
- natural language preprocessing techniques like **tokenization**, **stopword removal**, and **bag-of-words**
- the theory and implementation of the **Latent Dirichlet Allocation** topic model

It also reinforced my current skills in **data cleaning**, **machine learning models** , **grid search algorithms**, and **web application development**. 

For now, based on the dashboard, it looks like I need to learn some Tableau and PowerBI! 

Thank you for reading!

