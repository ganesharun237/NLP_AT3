

POS Tagging was performed on the Elon-Musk tweets dataset to extraxt only the nouns of tweets as it was proven from previous research that Topic Modelling on Nouns improves the Topic coherence. Once the nouns of Tweets was extracted Topic Modelling on the nouns of tweets was done using GSDMM model.



# GSDMM: Short text clustering

This project implements the Gibbs sampling algorithm for a Dirichlet Mixture Model of [Yin and Wang 2014](https://pdfs.semanticscholar.org/058a/d0815ce350f0e7538e00868c762be78fe5ef.pdf) for topic modelling of short text documents like Twitter tweets. 
Some advantages of this algorithm:
 - It requires only an upper bound `K` on the number of clusters
 - With good parameter selection, the model converges quickly
 - Space efficient and scalable

GSDMM model was selected and used because it performs better than the state of the art LDA model at Topic Modelling of short texts. LDA is better at Topic modelling of Large texts whereas GSDMM is better at Topic Modelling of short texts like Tweets.

## The Movie Group Process
In their paper, the authors introduce a simple conceptual model for explaining the GSDMM called the Movie Group Process.

Imagine a professor is leading a film class. At the start of the class, the students
are randomly assigned to `K` tables. Before class begins, the students make lists of
their favorite films. The professor repeatedly reads the class role. Each time the student's name is called,
the student must select a new table satisfying one or both of the following conditions:

- The new table has more students than the current table.
- The new table has students with similar lists of favorite movies.

By following these steps consistently, we might expect that the students eventually arrive at an "optimal" table configuration.

## Usage
To use a Movie Group Process to perform Topic Modelling of short texts, first initialize a [MovieGroupProcess](gsdmm/mgp.py):
```python
from gsdmm import MovieGroupProcess
mgp = MovieGroupProcess(K=8, alpha=0.1, beta=0.1, n_iters=30)
```
It's important to always choose `K` to be larger than the number of clusters you expect exist in your data, as the algorithm
can never return more than `K` clusters.

To fit the model:
```python
y = mgp.fit(docs)
```
Each doc in `docs` must be a unique list of tokens found in your short text document. This implementation does not support
counting tokens with multiplicity (which generally has little value in short text documents).
