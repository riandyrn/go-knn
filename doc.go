package knn

// Document represents single entry in KNN index
type Document interface {
	GetID() string
	GetVector() []float64
	GetDistance(vector []float64) (float64, error)
}

// ResultDocument is wrapper for Document but with
// extra information related to search result
// (e.g similarity distance)
type ResultDocument struct {
	Document Document
	Distance float64 // the lower the better
}
