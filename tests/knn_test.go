package test

import (
	"fmt"
	"math/rand"
	"strconv"
	"sync"
	"testing"

	"github.com/riandyrn/go-knn"
)

func TestAdd(t *testing.T) {
	testCases := []struct {
		Name      string
		InputDoc  knn.Document
		ExpErrNil bool
	}{
		{
			Name:      "Test Nil Document",
			InputDoc:  nil,
			ExpErrNil: false,
		},
		{
			Name:      "Test Empty ID",
			InputDoc:  newMockDoc("", []float64{1, 2, 3}),
			ExpErrNil: false,
		},
		{
			Name:      "Test Empty Vector",
			InputDoc:  newMockDoc("doc_1.xls", nil),
			ExpErrNil: false,
		},
		{
			Name:      "Test Normal Document",
			InputDoc:  newMockDoc("doc_1.xls", []float64{1, 2, 3}),
			ExpErrNil: true,
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.Name, func(t *testing.T) {
			knn := knn.NewKNN(knn.Configs{
				VectorDimension: 3,
				NumHashTable:    2,
				NumHyperplane:   3,
				SlotSize:        5,
			})
			err := knn.Add(testCase.InputDoc)
			if (err == nil) != testCase.ExpErrNil {
				t.Fatalf("unexpected error for case: %+v, err: %v", testCase, err)
			}
		})
	}
}

func TestQuery(t *testing.T) {
	// prepare documents
	docs := []*mockDoc{
		newMockDoc("doc_1", []float64{0, 0, 0, 0, 0}),
		newMockDoc("doc_2", []float64{0, 1, 0, 0, 0}),
		newMockDoc("doc_3", []float64{8, 6, 5, 4, 9}),
		newMockDoc("doc_4", []float64{0, 0, 2, 0, 1}),
		newMockDoc("doc_5", []float64{1, 2, 0, 0, 3}),
		newMockDoc("doc_6", []float64{0, 3, 1, 0, 0}),
	}
	// initialize knn index
	knn := knn.NewKNN(knn.Configs{
		VectorDimension: 5,
		NumHashTable:    3,
		NumHyperplane:   2,
		SlotSize:        5,
	})
	// insert document to knn
	for i := 0; i < len(docs); i++ {
		knn.Add(docs[i])
	}
	// prepare query vector
	queryVector := docs[0].GetVector()
	// execute query
	k := 2
	resultDocs, err := knn.Query(queryVector, k)
	if err != nil {
		t.Fatalf("unexpected error, err: %v", err)
	}
	// put similar docs id into map for fast lookup
	idMap := map[string]bool{}
	for _, resultDoc := range resultDocs {
		idMap[resultDoc.Document.GetID()] = true
	}
	// examine result
	expSimilarPostIDs := []string{"doc_1", "doc_2"}
	for _, id := range expSimilarPostIDs {
		_, ok := idMap[id]
		if !ok {
			t.Fatalf("document with id: %v is not found on result", id)
		}
	}
}

func TestDelete(t *testing.T) {
	dim := 100
	testCases := []struct {
		Name            string
		InitialDocs     []knn.Document
		DocID           string
		ExpDeleteErrNil bool
	}{
		{
			Name: "Test Empty ID",
			InitialDocs: []knn.Document{
				newMockDoc("doc_1", getRandomVector(dim)),
				newMockDoc("doc_2", getRandomVector(dim)),
			},
			DocID:           "",
			ExpDeleteErrNil: false,
		},
		{
			Name: "Test Delete Non-Existing Document",
			InitialDocs: []knn.Document{
				newMockDoc("doc_1", getRandomVector(dim)),
				newMockDoc("doc_2", getRandomVector(dim)),
			},
			DocID:           "doc_3",
			ExpDeleteErrNil: true,
		},
		{
			Name: "Test Delete Existing Document",
			InitialDocs: []knn.Document{
				newMockDoc("doc_1", getRandomVector(dim)),
				newMockDoc("doc_2", getRandomVector(dim)),
			},
			DocID:           "doc_1",
			ExpDeleteErrNil: true,
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.Name, func(t *testing.T) {
			// initialize knn index
			knn := knn.NewKNN(knn.Configs{
				VectorDimension: dim,
				NumHashTable:    1,
				NumHyperplane:   2,
				SlotSize:        3,
			})
			// insert sample docs
			for _, doc := range testCase.InitialDocs {
				knn.Add(doc)
			}
			// delete doc
			err := knn.Delete(testCase.DocID)
			if (err == nil) != testCase.ExpDeleteErrNil {
				t.Fatalf("unexpected error, for testcase: %+v, got: %v", testCase, err)
			}
			if !testCase.ExpDeleteErrNil {
				return
			}
			// try to get the deleted doc
			doc, err := knn.Get(testCase.DocID)
			if err != nil {
				t.Fatalf("unexpected error when fetching document from index for testcase: %+v, got: %v", testCase, err)
			}
			if doc != nil {
				t.Fatalf("document %v still found on index", doc.GetID())
			}
		})
	}
}

func TestConcurrentQueries(t *testing.T) {
	// prepare documents
	n := 10000
	dim := 100
	documents := getMockDocuments(n, dim)
	// prepare knn index
	knn := knn.NewKNN(knn.Configs{
		VectorDimension: dim,
		NumHashTable:    10,
		NumHyperplane:   10,
		SlotSize:        20,
	})
	// insert document to index
	for _, document := range documents {
		knn.Add(document)
	}
	// execute multiple goroutine to call query concurrently
	numGoroutines := 200
	k := 5
	var wg sync.WaitGroup
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			queryVector := getRandomVector(dim)
			knn.Query(queryVector, k)
		}()
	}
	wg.Wait()
}

func TestConcurrentRWOps(t *testing.T) {
	// prepare documents
	n := 10000
	dim := 100
	documents := getMockDocuments(n, dim)
	// prepare knn index
	knn := knn.NewKNN(knn.Configs{
		VectorDimension: dim,
		NumHashTable:    10,
		NumHyperplane:   10,
		SlotSize:        20,
	})
	// insert initial document to index
	for _, document := range documents {
		knn.Add(document)
	}
	// execute multiple goroutine to call multiple ops concurrently
	numGoroutines := 1000
	k := 5
	var wg sync.WaitGroup
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()

			switch i % 3 {
			case 0:
				queryVector := getRandomVector(dim)
				knn.Query(queryVector, k)
			case 1:
				doc := newMockDoc(
					fmt.Sprintf("doc_%v", i),
					getRandomVector(dim),
				)
				knn.Add(doc)
			case 2:
				knn.Delete(strconv.Itoa(rand.Intn(n)))
			}

		}(i)
	}
	wg.Wait()
}
