package main

import (
	"log"

	"github.com/riandyrn/go-knn"
)

func main() {
	// prepare knn index
	knn := knn.NewKNN(knn.Configs{
		VectorDimension: 5,
		NumHashTable:    3,
		NumHyperplane:   3,
		SlotSize:        5,
	})
	// prepare documents
	imageDocs := []ImageDoc{
		{
			ID:     "image_1.jpeg",
			Vector: []float64{0, 0, 0, 0, 0},
		},
		{
			ID:     "image_2.jpeg",
			Vector: []float64{0, 1, 0, 0, 0},
		},
		{
			ID:     "image_3.jpeg",
			Vector: []float64{8, 6, 5, 4, 9},
		},
		{
			ID:     "image_4.jpeg",
			Vector: []float64{0, 0, 2, 0, 1},
		},
		{
			ID:     "image_5.jpeg",
			Vector: []float64{1, 2, 0, 0, 3},
		},
		{
			ID:     "image_6.jpeg",
			Vector: []float64{0, 0, 1, 0, 0},
		},
	}
	// add documents to knn index
	for i := 0; i < len(imageDocs); i++ {
		err := knn.Add(&imageDocs[i])
		if err != nil {
			log.Fatalf("unable to add new document to knn index due: %v", err)
		}
	}

	// query documents
	queryVector := imageDocs[0].GetVector()
	log.Printf("query vector: %v", queryVector)
	resultDocs, err := knn.Query(queryVector, 2)
	if err != nil {
		log.Fatalf("unable to query documents due: %v", err)
	}
	for _, resultDoc := range resultDocs {
		log.Printf("similar doc: %+v, distance: %v", resultDoc.Document, resultDoc.Distance)
	}

	// delete document from result
	knn.Delete(resultDocs[0].Document.GetID())
	log.Printf("successfully delete document with id: %v", resultDocs[0].Document.GetID())

	// re-query documents
	log.Printf("query vector: %v", queryVector)
	resultDocs, err = knn.Query(queryVector, 2)
	if err != nil {
		log.Fatalf("unable to query documents due: %v", err)
	}
	for _, resultDoc := range resultDocs {
		log.Printf("similar doc: %+v, distance: %v", resultDoc.Document, resultDoc.Distance)
	}
}
