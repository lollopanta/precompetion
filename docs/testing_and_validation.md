# BloomTrack: Testing and Validation Strategies

This document outlines recommended testing approaches, validation metrics, and quality assurance strategies for the BloomTrack system.

## 1. Unit Testing

### Data Ingestion and Preprocessing

**Test Components:**
- `LandsatAcquisition` class
- Vegetation indices calculation functions
- Environmental feature extraction

**Recommended Tests:**

```python
# Example unit test for NDVI calculation
def test_ndvi_calculation():
    # Create synthetic NIR and Red bands
    nir = np.array([[0.5, 0.4], [0.3, 0.6]])
    red = np.array([[0.2, 0.3], [0.2, 0.1]])
    
    # Calculate NDVI
    ndvi_result = calculate_ndvi(nir, red)
    
    # Expected NDVI values calculated manually
    expected = np.array([[0.428571, 0.142857], [0.2, 0.714286]])
    
    # Assert that the calculated values match expected values within tolerance
    np.testing.assert_allclose(ndvi_result, expected, rtol=1e-5)
```

```python
# Example unit test for environmental feature extraction
def test_zonal_statistics():
    # Create a simple raster and polygon
    raster_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    transform = rasterio.transform.from_origin(0, 0, 1, 1)
    geometry = shapely.geometry.box(0.5, 0.5, 2.5, 2.5)
    
    # Calculate zonal statistics
    stats = zonal_statistics(raster_data, geometry, transform)
    
    # Expected statistics for the given geometry (covers cells with values 5, 6, 8, 9)
    assert stats['mean'] == 7.0
    assert stats['min'] == 5.0
    assert stats['max'] == 9.0
```

### Models

**Test Components:**
- `BloomClassifier`
- `BloomForecaster`
- `SpatialBloomModel`
- `BloomPredictor`

**Recommended Tests:**

```python
# Example unit test for BloomClassifier
def test_bloom_classifier_forward_pass():
    # Create a simple model instance
    model = BloomClassifier(input_dim=10, hidden_dim=8)
    
    # Create a dummy input tensor
    x = torch.randn(5, 10)  # Batch size 5, feature dimension 10
    
    # Perform forward pass
    output = model(x)
    
    # Check output shape and type
    assert output.shape == (5, 1)
    assert isinstance(output, torch.Tensor)
```

```python
# Example unit test for BloomPredictor
def test_bloom_predictor_load():
    # Create mock model files
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create dummy model files
        torch.save({'state_dict': {}}, os.path.join(tmp_dir, 'classifier.pt'))
        torch.save({'state_dict': {}}, os.path.join(tmp_dir, 'forecaster.pt'))
        
        # Create a dummy spatial model
        spatial_model = MockSpatialModel()
        with open(os.path.join(tmp_dir, 'spatial_model.pkl'), 'wb') as f:
            pickle.dump(spatial_model, f)
        
        # Test loading the predictor
        predictor = load_predictor(model_dir=tmp_dir)
        
        # Assert that the predictor has loaded all models
        assert predictor.classifier is not None
        assert predictor.forecaster is not None
        assert predictor.spatial_model is not None
```

### API

**Test Components:**
- API endpoints
- Request validation
- Response formatting

**Recommended Tests:**

```python
# Example API test using FastAPI TestClient
def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
```

```python
# Example API test for bloom prediction endpoint
def test_predict_bloom_endpoint():
    client = TestClient(app)
    
    # Test with valid coordinates
    response = client.post(
        "/api/predict/location",
        json={"latitude": 37.7749, "longitude": -122.4194, "date": "2023-06-15"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "bloom_probability" in data
    assert "forecast" in data
    
    # Test with invalid coordinates (out of range)
    response = client.post(
        "/api/predict/location",
        json={"latitude": 100.0, "longitude": -122.4194, "date": "2023-06-15"}
    )
    assert response.status_code == 422  # Validation error
```

### Frontend

**Test Components:**
- Map initialization
- API client
- UI interactions

**Recommended Tests:**

```javascript
// Example Jest test for BloomTrackAPI
describe('BloomTrackAPI', () => {
  let api;
  
  beforeEach(() => {
    // Mock fetch
    global.fetch = jest.fn();
    api = new BloomTrackAPI('http://localhost:8000');
  });
  
  test('getBloomPredictionForLocation calls correct endpoint', async () => {
    // Setup mock response
    global.fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ bloom_probability: 0.75 })
    });
    
    // Call the method
    const result = await api.getBloomPredictionForLocation(37.7749, -122.4194, '2023-06-15');
    
    // Verify fetch was called with correct arguments
    expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:8000/api/predict/location',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          latitude: 37.7749,
          longitude: -122.4194,
          date: '2023-06-15'
        })
      })
    );
    
    // Verify result
    expect(result.bloom_probability).toBe(0.75);
  });
});
```

## 2. Integration Testing

### Data Pipeline Integration

**Test Scenarios:**
- End-to-end data flow from acquisition to feature extraction
- Integration between data preprocessing and model input preparation

**Recommended Tests:**

```python
# Example integration test for data pipeline
def test_data_pipeline_integration():
    # Test parameters
    lat, lon = 37.7749, -122.4194
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    
    # Initialize components
    landsat = LandsatAcquisition(api_key="test_key")
    env_extractor = EnvironmentalFeatureExtractor()
    
    # Execute pipeline
    scenes = landsat.search_scenes(lat, lon, start_date, end_date)
    assert len(scenes) > 0, "No scenes found"
    
    scene_data = landsat.load_scene_data(scenes[0])
    assert "nir" in scene_data and "red" in scene_data, "Required bands not found"
    
    ndvi = calculate_ndvi(scene_data["nir"], scene_data["red"])
    assert ndvi is not None, "NDVI calculation failed"
    
    point_geom = shapely.geometry.Point(lon, lat).buffer(0.01)
    env_features = env_extractor.extract_features(point_geom)
    assert "elevation" in env_features, "Environmental features extraction failed"
    
    # Combine features for model input
    features = {**{"ndvi": ndvi.mean()}, **env_features}
    assert len(features) >= 4, "Insufficient features for model input"
```

### Model Pipeline Integration

**Test Scenarios:**
- Data preprocessing → Model training → Evaluation
- Model inference pipeline

**Recommended Tests:**

```python
# Example integration test for model pipeline
def test_model_training_pipeline():
    # Create synthetic dataset
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, size=100)
    X_val = np.random.rand(20, 10)
    y_val = np.random.randint(0, 2, size=20)
    
    # Convert to PyTorch datasets
    train_dataset = BloomDataset(X_train, y_train)
    val_dataset = BloomDataset(X_val, y_val)
    
    # Initialize model and trainer
    model = BloomClassifier(input_dim=10, hidden_dim=8)
    trainer = BloomClassifierTrainer(model)
    
    # Train model
    history = trainer.train(train_dataset, val_dataset, epochs=5)
    
    # Verify training occurred
    assert len(history['train_loss']) == 5, "Training did not complete"
    assert len(history['val_loss']) == 5, "Validation did not complete"
    
    # Test prediction
    predictions = trainer.predict(X_val)
    assert predictions.shape == (20,), "Predictions shape mismatch"
```

### API and Frontend Integration

**Test Scenarios:**
- API endpoints with actual model inference
- Frontend components with API integration

**Recommended Tests:**

```python
# Example API integration test
def test_api_model_integration():
    # Create a test client with a real model instance
    app.dependency_overrides[get_bloom_predictor] = lambda: create_test_predictor()
    client = TestClient(app)
    
    # Test prediction endpoint
    response = client.post(
        "/api/predict/location",
        json={"latitude": 37.7749, "longitude": -122.4194, "date": "2023-06-15"}
    )
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["bloom_probability"] <= 1, "Invalid probability value"
    assert len(data["forecast"]) > 0, "No forecast data returned"
```

## 3. System Testing

### Performance Testing

**Test Scenarios:**
- API response time under load
- Batch processing performance
- Map rendering performance with multiple layers

**Recommended Tools:**
- Locust or JMeter for load testing
- Prometheus and Grafana for monitoring

**Example Test Plan:**

```python
# Example Locust load test
from locust import HttpUser, task, between

class BloomTrackUser(HttpUser):
    wait_time = between(1, 5)
    
    @task(3)
    def get_location_prediction(self):
        # Randomly select coordinates
        lat = random.uniform(25, 50)
        lon = random.uniform(-125, -65)
        date = "2023-06-15"
        
        self.client.post(
            "/api/predict/location",
            json={"latitude": lat, "longitude": lon, "date": date}
        )
    
    @task(1)
    def get_bbox_prediction(self):
        # Random bounding box (small area)
        min_lat = random.uniform(25, 49)
        min_lon = random.uniform(-125, -66)
        max_lat = min_lat + 1.0
        max_lon = min_lon + 1.0
        date = "2023-06-15"
        
        self.client.post(
            "/api/predict/bbox",
            json={
                "min_lat": min_lat, "min_lon": min_lon,
                "max_lat": max_lat, "max_lon": max_lon,
                "date": date
            }
        )
```

### End-to-End Testing

**Test Scenarios:**
- Complete user workflows
- Data ingestion to visualization

**Recommended Approach:**
- Cypress or Selenium for frontend testing
- Automated scripts for backend workflows

**Example Test:**

```javascript
// Example Cypress end-to-end test
describe('BloomTrack Map Interaction', () => {
  beforeEach(() => {
    cy.visit('http://localhost:3000');
    cy.wait(2000); // Wait for map to load
  });
  
  it('should display bloom prediction when location is clicked', () => {
    // Click on map at specific coordinates
    cy.get('#map').click(300, 200);
    
    // Wait for API call to complete
    cy.wait('@getPrediction');
    
    // Verify popup appears with prediction data
    cy.get('.leaflet-popup').should('be.visible');
    cy.get('.leaflet-popup-content').should('contain', 'Bloom Probability');
    
    // Verify chart is displayed
    cy.get('.forecast-chart').should('be.visible');
  });
  
  it('should update map when date is changed', () => {
    // Change date in date picker
    cy.get('#date-picker').type('2023-07-15');
    cy.get('#update-button').click();
    
    // Wait for API call to complete
    cy.wait('@getBboxPrediction');
    
    // Verify map layer is updated
    cy.get('.leaflet-tile-loaded').should('have.length.at.least', 1);
  });
});
```

## 4. Validation Metrics

### Model Validation

#### Classification Metrics

- **Accuracy:** Overall correctness of bloom classification
- **Precision:** Proportion of positive identifications that were actually correct
- **Recall:** Proportion of actual positives that were identified correctly
- **F1 Score:** Harmonic mean of precision and recall
- **AUC-ROC:** Area under the Receiver Operating Characteristic curve

**Implementation Example:**

```python
def evaluate_bloom_classifier(model, test_dataset):
    """Evaluate bloom classifier using multiple metrics."""
    y_true = []
    y_pred = []
    y_prob = []
    
    # Get predictions
    for X, y in test_dataset:
        outputs = model(X)
        probs = torch.sigmoid(outputs).detach().numpy()
        preds = (probs > 0.5).astype(int)
        
        y_true.extend(y.numpy())
        y_pred.extend(preds)
        y_prob.extend(probs)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc,
        'confusion_matrix': cm
    }
```

#### Forecasting Metrics

- **Mean Absolute Error (MAE):** Average absolute difference between predicted and actual values
- **Root Mean Square Error (RMSE):** Square root of the average squared differences
- **Mean Absolute Percentage Error (MAPE):** Average percentage difference
- **R-squared:** Proportion of variance explained by the model

**Implementation Example:**

```python
def evaluate_bloom_forecaster(model, test_dataset, forecast_horizon):
    """Evaluate bloom forecaster using multiple metrics."""
    y_true = []
    y_pred = []
    
    # Get predictions
    for X, y in test_dataset:
        outputs = model(X)
        y_true.extend(y.numpy())
        y_pred.extend(outputs.detach().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics for each forecast step
    metrics = {}
    for i in range(forecast_horizon):
        true_i = y_true[:, i]
        pred_i = y_pred[:, i]
        
        mae = mean_absolute_error(true_i, pred_i)
        rmse = np.sqrt(mean_squared_error(true_i, pred_i))
        
        # Calculate MAPE, avoiding division by zero
        mask = true_i != 0
        mape = np.mean(np.abs((true_i[mask] - pred_i[mask]) / true_i[mask])) * 100
        
        r2 = r2_score(true_i, pred_i)
        
        metrics[f'step_{i+1}'] = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }
    
    # Calculate overall metrics
    metrics['overall'] = {
        'mae': mean_absolute_error(y_true.flatten(), y_pred.flatten()),
        'rmse': np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten())),
        'r2': r2_score(y_true.flatten(), y_pred.flatten())
    }
    
    return metrics
```

#### Spatial Model Metrics

- **Spatial Autocorrelation (Moran's I):** Measure of spatial dependence
- **RMSE with Cross-Validation:** Error using spatial cross-validation
- **Area Under Precision-Recall Curve:** For imbalanced spatial classification

**Implementation Example:**

```python
def evaluate_spatial_model(model, X, y, geometries):
    """Evaluate spatial model using spatial metrics."""
    # Predict values
    y_pred = model.predict(X)
    
    # Calculate standard metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    # Calculate residuals
    residuals = y - y_pred
    
    # Create spatial weights matrix
    from pysal.lib import weights
    w = weights.distance.DistanceBand.from_array(np.array([(g.centroid.x, g.centroid.y) for g in geometries]))
    
    # Calculate Moran's I for residuals
    from esda.moran import Moran
    moran = Moran(residuals, w)
    
    return {
        'rmse': rmse,
        'r2': r2,
        'morans_i': moran.I,
        'morans_p_value': moran.p_sim
    }
```

### API Validation

- **Response Time:** Time to respond to requests (95th percentile, median)
- **Error Rate:** Percentage of requests resulting in errors
- **Throughput:** Requests handled per second

### Frontend Validation

- **Time to Interactive:** Time until the map is fully interactive
- **Frame Rate:** During map interactions (panning, zooming)
- **Memory Usage:** Browser memory consumption

## 5. Continuous Integration and Testing

### CI/CD Pipeline

**Recommended Setup:**
- GitHub Actions or Jenkins for automation
- Test execution on pull requests
- Automated deployment with staging environment

**Example GitHub Actions Workflow:**

```yaml
name: BloomTrack CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest pytest-cov
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
    - name: Run tests
      run: |
        pytest --cov=backend tests/
    - name: Upload coverage report
      uses: codecov/codecov-action@v1

  frontend-test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Node.js
      uses: actions/setup-node@v2
      with:
        node-version: '16'
    - name: Install dependencies
      run: |
        cd frontend
        npm install
    - name: Run tests
      run: |
        cd frontend
        npm test
```

### Test Coverage Goals

- Backend code: 80%+ coverage
- API endpoints: 100% coverage
- Critical model components: 90%+ coverage
- Frontend components: 70%+ coverage

## 6. Data Quality Validation

### Input Data Validation

- **Completeness:** Check for missing data in satellite imagery
- **Consistency:** Validate temporal consistency of time series
- **Range Validation:** Ensure values are within expected ranges

**Implementation Example:**

```python
def validate_satellite_data(dataset):
    """Validate quality of satellite imagery dataset."""
    validation_results = {
        'passed': True,
        'issues': []
    }
    
    # Check for missing data
    missing_percentage = np.isnan(dataset).sum() / dataset.size * 100
    if missing_percentage > 10:
        validation_results['passed'] = False
        validation_results['issues'].append(f"High missing data: {missing_percentage:.2f}%")
    
    # Check for out-of-range values (example for reflectance data)
    if np.any(dataset < 0) or np.any(dataset > 1):
        validation_results['passed'] = False
        validation_results['issues'].append("Values outside expected range [0, 1]")
    
    # Check for cloud coverage (if cloud mask is available)
    if 'cloud_mask' in dataset.attrs:
        cloud_percentage = dataset.attrs['cloud_mask'].mean() * 100
        if cloud_percentage > 30:
            validation_results['passed'] = False
            validation_results['issues'].append(f"High cloud coverage: {cloud_percentage:.2f}%")
    
    return validation_results
```

### Output Validation

- **Spatial Coherence:** Check for unrealistic spatial patterns
- **Temporal Consistency:** Validate smooth temporal transitions
- **Edge Case Handling:** Test with extreme environmental conditions

## 7. Monitoring and Alerting

### Runtime Monitoring

- **Model Drift:** Monitor prediction distribution changes
- **Data Drift:** Monitor input feature distribution changes
- **Performance Metrics:** Track API response times and error rates

**Implementation Example:**

```python
def monitor_model_drift(predictions, reference_distribution):
    """Monitor for drift in model predictions."""
    # Calculate current distribution statistics
    current_mean = np.mean(predictions)
    current_std = np.std(predictions)
    
    # Compare with reference distribution
    mean_diff = abs(current_mean - reference_distribution['mean'])
    std_diff = abs(current_std - reference_distribution['std'])
    
    # Check for significant drift
    mean_threshold = 0.1
    std_threshold = 0.2
    
    if mean_diff > mean_threshold or std_diff > std_threshold:
        # Alert on drift detected
        return {
            'drift_detected': True,
            'mean_diff': mean_diff,
            'std_diff': std_diff
        }
    
    return {'drift_detected': False}
```

### Alerting System

- **Error Rate Alerts:** Notify when error rates exceed thresholds
- **Performance Degradation:** Alert on slow response times
- **Model Drift Alerts:** Notify when predictions show significant drift

## Conclusion

A comprehensive testing and validation strategy is essential for ensuring the reliability, accuracy, and performance of the BloomTrack system. By implementing the recommended tests and validation metrics, the system can be continuously monitored and improved to provide accurate bloom predictions and a responsive user experience.

The testing strategy should evolve alongside the system, with new tests added as features are developed and existing tests refined based on real-world usage patterns and feedback.