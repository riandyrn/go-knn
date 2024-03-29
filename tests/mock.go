package test

import (
	"fmt"
	"math/rand"

	"github.com/riandyrn/go-knn"
)

func newMockDoc(id string, vector []float64) *mockDoc {
	return &mockDoc{
		id:     id,
		vector: vector,
	}
}

type mockDoc struct {
	id     string
	vector []float64
}

func (d *mockDoc) GetID() string { return d.id }

func (d *mockDoc) GetVector() []float64 { return d.vector }

func getRandomVector(dim int) []float64 {
	vector := make([]float64, 0, dim)
	for j := 0; j < dim; j++ {
		vector = append(vector, rand.NormFloat64())
	}
	return vector
}

func getMockDocuments(n, dim int) []knn.Document {
	documents := make([]knn.Document, 0, n)
	for i := 0; i < n; i++ {
		id := fmt.Sprintf("doc_%v", i)
		vector := getRandomVector(dim)
		documents = append(documents, newMockDoc(id, vector))
	}
	return documents
}
