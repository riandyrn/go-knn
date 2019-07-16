package main

import (
	"fmt"
	"math"
)

// ImageDoc represents single image document
type ImageDoc struct {
	ID     string
	Vector []float64
}

// GetID returns identifier of ImageDoc
func (d *ImageDoc) GetID() string { return d.ID }

// GetVector returns vector of ImageDoc
func (d *ImageDoc) GetVector() []float64 { return d.Vector }

// GetDistance returns distance of internal vector to another vector,
// we use euclidean distance to calculate it
func (d *ImageDoc) GetDistance(vector []float64) (float64, error) {
	if len(d.Vector) != len(vector) {
		return 0, fmt.Errorf("unable to get distance since input vector has different dimension")
	}
	sum := 0.0
	for i := 0; i < len(d.Vector); i++ {
		sum += math.Pow(float64(d.Vector[i]-vector[i]), 2.0)
	}
	distance := math.Sqrt(sum)

	return distance, nil
}
