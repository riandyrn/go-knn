package knn

// Configs holds configuration for KNN
type Configs struct {
	// VectorDimension represents expected number of
	// dimension in the input vector. For example vector
	// [a, b, c, d, e] has dimension 5.
	VectorDimension int

	// NumHashTable represents number of hash tables exists
	// on memory. The usage of multiple tables is to lower
	// the index chance to miss near duplicates. Checkout
	// https://www.youtube.com/watch?v=dgH0NP8Qxa8
	// for more detailed explanation.
	NumHashTable int

	// NumHyperplane represents number of hyperplane exists in single
	// hash table is used to classify where the documents would be
	// placed on hash table. So in essence this parameter define the
	// hash key for input document in the table. For more details
	// checkout https://www.youtube.com/watch?v=Arni-zkqMBA.
	NumHyperplane int

	// SlotSize represents the slot size for each locus in hyperplane,
	// essentially it is used as "tolerance" degree for the input document.
	// Settings the value of SlotSize is tricky & might differ from case
	// to case, so it is necessary to experiment a bit with your data.
	// For vector with 512 dimension, try to use value `40`.
	//
	// Checkout https://github.com/ekzhu/lsh/issues/2 for details. This
	// parameter refers to `w` parameter mentioned on the issue.
	SlotSize int
}
