package main

// ImageDoc represents single image document
type ImageDoc struct {
	ID     string
	Vector []float64
}

// GetID returns identifier of ImageDoc
func (d *ImageDoc) GetID() string { return d.ID }

// GetVector returns vector of ImageDoc
func (d *ImageDoc) GetVector() []float64 { return d.Vector }
