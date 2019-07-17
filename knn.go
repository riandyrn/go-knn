package knn

import (
	"fmt"
	"math"
	"sort"
	"sync"

	"github.com/riandyrn/lsh"
)

// KNN is the index for searching nearest neighbors.
// It is based on LSH index.
type KNN struct {
	// LSH index which will be used for indexing documents
	lsh *lsh.BasicLsh

	// We also store value of vector dimension because we
	// still need it for input validation
	vectorDimension int

	// We use another map because LSH index only stores document
	// id, so to get full information of document we need another
	// map for it.
	//
	// The behavior of LSH index is pretty reasonable, because in
	// the index, the document would be stored in all hash tables
	// (duplicated). So it's best to just store the necessary info
	// on the LSH index to minimize memory usage, which in this
	// case is document id.
	docMap sync.Map

	// We use mutex because the LSH implementation used
	// normal map instead of sync map, yet we are expecting
	// to use the LSH concurrently for read & write. So mutex
	// is needed to prevent panic from map.
	mux sync.RWMutex
}

// NewKNN returns new initialized instance of KNN
// index. It uses basic LSH algorithm implemented on
// `github.com/ekzhu/lsh` as its engine to search for
// nearest neighbors.
func NewKNN(configs Configs) *KNN {
	return &KNN{
		vectorDimension: configs.VectorDimension,
		lsh: lsh.NewBasicLsh(
			configs.VectorDimension,
			configs.NumHashTable,
			configs.NumHyperplane,
			float64(configs.SlotSize),
		),
		docMap: sync.Map{},
	}
}

// Add is used for introduce new document to index
func (n *KNN) Add(doc Document) error {
	// check input validity
	if doc == nil || len(doc.GetID()) == 0 || len(doc.GetVector()) == 0 {
		return fmt.Errorf("trying to insert bad document")
	}
	dim := len(doc.GetVector())
	if dim != n.vectorDimension {
		return fmt.Errorf("unexpected vector dimension, expected: %v, got: %v", n.vectorDimension, dim)
	}
	// acquire lock
	n.mux.Lock()
	// defer unlock
	defer n.mux.Unlock()

	// insert document to LSH index
	n.lsh.Insert(doc.GetVector(), doc.GetID())
	// insert document to map
	n.docMap.Store(doc.GetID(), doc)

	return nil
}

// Query returns maximum `k` similar documents. The result
// already sorted from most similar to least similar documents.
func (n *KNN) Query(vector []float64, k int) ([]ResultDocument, error) {
	// check input validity
	if len(vector) == 0 {
		return nil, fmt.Errorf("vector must not empty")
	}
	if k <= 0 {
		return nil, fmt.Errorf("value of k must be greater than 0")
	}
	// acquire read lock
	n.mux.RLock()
	// defer read unlock
	defer n.mux.RUnlock()

	// get ids of similar documents
	ids := n.lsh.Query(vector)
	// get full document info from docMap including
	// distance from input vector
	resultDocs := make([]ResultDocument, 0, len(ids))
	for _, id := range ids {
		v, ok := n.docMap.Load(id)
		if !ok {
			continue
		}
		doc := v.(Document)
		distance := calcDistance(doc.GetVector(), vector)
		resultDocs = append(resultDocs, ResultDocument{
			Document: doc,
			Distance: distance,
		})
	}
	// sort by distance from minimum to maximum
	sort.Slice(resultDocs, func(i int, j int) bool {
		return resultDocs[i].Distance < resultDocs[j].Distance
	})
	// cut the result into max k documents
	if len(resultDocs) > k {
		resultDocs = resultDocs[:k]
	}
	return resultDocs, nil
}

// calcDistance is used for calculating vector distance using
// euclidean formula. Input `v1` & `v2` assummed has same dimension.
func calcDistance(v1, v2 []float64) float64 {
	sum := 0.0
	for i := 0; i < len(v1); i++ {
		sum += math.Pow(float64(v2[i]-v1[i]), 2.0)
	}
	distance := math.Sqrt(sum)

	return distance
}

// Delete is used to delete appointed document from index.
// The document will literally deleted from memory.
func (n *KNN) Delete(docID string) error {
	// check input validity
	if len(docID) == 0 {
		return fmt.Errorf("document id must not empty")
	}
	// acquire lock
	n.mux.Lock()
	// defer unlock
	defer n.mux.Unlock()

	// delete from lsh index
	n.lsh.Delete(docID)
	// delete document from map
	n.docMap.Delete(docID)

	return nil
}

// Get is used to get single document from index.
// If document not found returns nil instead.
func (n *KNN) Get(docID string) (Document, error) {
	// check input validity
	if len(docID) == 0 {
		return nil, fmt.Errorf("document id must not empty")
	}
	// load document from doc map
	v, ok := n.docMap.Load(docID)
	if !ok {
		return nil, nil
	}
	return v.(Document), nil
}
