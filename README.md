# Go KNN

[![Build Status](https://travis-ci.org/riandyrn/go-knn.svg?branch=master)](https://travis-ci.org/riandyrn/go-knn)
[![Go Report Card](https://goreportcard.com/badge/github.com/riandyrn/go-knn)](https://goreportcard.com/report/github.com/riandyrn/go-knn)

KNN In-Memory Index for Go. Extension of Basic LSH Algorithm implemented by [@ekzhu](https://github.com/ekzhu/lsh).

For sample usage, checkout `/example` dir

**Added Features:**

- Index is safe to be accessed concurrently
- User could put whole document in the index, not only its id
- Added true distance comparison for documents inside the bucket to eliminate false positives

**Great Resources:**

- [LSH.8 Locality-sensitive hashing: the idea](https://www.youtube.com/watch?v=dgH0NP8Qxa8)
- [LSH.9 Locality-sensitive hashing: how it works](https://www.youtube.com/watch?v=Arni-zkqMBA)
- [LSH.10 False positive and negative errors of LSH](https://www.youtube.com/watch?v=h21irtHDsBw)

---
