package knn

// Document represents single entry in KNN index
type Document interface {
	GetID() string
	GetVector() []float64
}

// ResultDocument is wrapper for Document but with
// extra information related to search result
// (e.g similarity distance)
type ResultDocument struct {
	Document Document
	Distance float64 // the lower the better
}
