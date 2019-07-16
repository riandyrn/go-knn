package test

import (
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
			InputDoc:  newMockDoc("", []float64{1, 2, 3}, 2),
			ExpErrNil: false,
		},
		{
			Name:      "Test Empty Vector",
			InputDoc:  newMockDoc("doc_1.xls", nil, 2),
			ExpErrNil: false,
		},
		{
			Name:      "Test Normal Document",
			InputDoc:  newMockDoc("doc_1.xls", []float64{1, 2, 3}, 2),
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

func TestQuery(t *testing.T) {}

func TestDelete(t *testing.T) {}

func TestConcurrentQueries(t *testing.T) {}

func TestConcurrentRWOps(t *testing.T) {}
