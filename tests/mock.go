package test

func newMockDoc(id string, vector []float64, distance float64) *mockDoc {
	return &mockDoc{
		id:       id,
		vector:   vector,
		distance: distance,
	}
}

type mockDoc struct {
	id       string
	vector   []float64
	distance float64
}

func (d *mockDoc) GetID() string { return d.id }

func (d *mockDoc) GetVector() []float64 { return d.vector }

func (d *mockDoc) GetDistance(vector []float64) (float64, error) { return d.distance, nil }
