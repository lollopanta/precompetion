# BloomTrack Project Structure

```
bloomtrack/
├── backend/
│   ├── data/
│   │   ├── acquisition/       # Data acquisition from satellite sources
│   │   │   ├── __init__.py
│   │   │   ├── landsat.py     # Landsat-specific data acquisition
│   │   │   ├── modis.py       # MODIS-specific data acquisition
│   │   │   └── viirs.py       # VIIRS-specific data acquisition
│   │   ├── preprocessing/     # Data preprocessing and cleaning
│   │   │   ├── __init__.py
│   │   │   ├── alignment.py   # Spatial alignment utilities
│   │   │   ├── conversion.py  # Format conversion utilities
│   │   │   ├── indices.py     # Vegetation indices calculation
│   │   │   └── quality.py     # Quality control and filtering
│   │   ├── features/          # Feature extraction and engineering
│   │   │   ├── __init__.py
│   │   │   ├── environmental.py  # Environmental feature extraction
│   │   │   ├── spectral.py    # Spectral feature extraction
│   │   │   └── temporal.py    # Temporal feature extraction
│   │   └── __init__.py
│   ├── models/
│   │   ├── training/          # Model training pipelines
│   │   │   ├── __init__.py
│   │   │   ├── bloom_classifier.py  # Bloom classification model
│   │   │   ├── bloom_regressor.py   # Bloom intensity regression model
│   │   │   └── temporal_model.py    # Temporal prediction model
│   │   ├── inference/         # Model inference services
│   │   │   ├── __init__.py
│   │   │   ├── batch_predictor.py   # Batch prediction service
│   │   │   └── realtime_predictor.py # Real-time prediction service
│   │   ├── evaluation/        # Model evaluation and metrics
│   │   │   ├── __init__.py
│   │   │   ├── metrics.py     # Evaluation metrics
│   │   │   └── validation.py  # Validation utilities
│   │   └── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI application entry point
│   │   ├── auth.py            # Authentication and authorization
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── predictions.py # Prediction API endpoints
│   │   │   ├── data.py        # Data access API endpoints
│   │   │   └── tiles.py       # Map tile API endpoints
│   │   ├── schemas/           # Pydantic schemas for API
│   │   │   ├── __init__.py
│   │   │   ├── requests.py    # Request schemas
│   │   │   └── responses.py   # Response schemas
│   │   └── dependencies.py    # FastAPI dependencies
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── database.py        # Database connection and models
│   │   ├── raster_store.py    # Raster data storage utilities
│   │   ├── vector_store.py    # Vector data storage utilities
│   │   └── object_store.py    # Object storage utilities
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── worker.py          # Celery worker configuration
│   │   ├── data_tasks.py      # Data processing tasks
│   │   └── prediction_tasks.py # Prediction tasks
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration management
│   │   ├── logging.py         # Logging utilities
│   │   └── geo_utils.py       # Geospatial utilities
│   └── __init__.py
├── frontend/
│   ├── public/
│   │   ├── index.html
│   │   ├── favicon.ico
│   │   └── assets/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Map/
│   │   │   │   ├── MapContainer.jsx  # Main map component
│   │   │   │   ├── LayerControl.jsx  # Layer control component
│   │   │   │   └── TimeSlider.jsx    # Time slider component
│   │   │   ├── Dashboard/
│   │   │   │   ├── Dashboard.jsx     # Main dashboard component
│   │   │   │   ├── BloomChart.jsx    # Bloom statistics chart
│   │   │   │   └── LocationDetail.jsx # Location detail component
│   │   │   ├── Common/
│   │   │   │   ├── Header.jsx        # Header component
│   │   │   │   ├── Footer.jsx        # Footer component
│   │   │   │   └── Sidebar.jsx       # Sidebar component
│   │   │   └── index.js
│   │   ├── services/
│   │   │   ├── api.js          # API client
│   │   │   ├── mapService.js   # Map service utilities
│   │   │   └── authService.js  # Authentication service
│   │   ├── hooks/
│   │   │   ├── useMap.js       # Map hook
│   │   │   ├── useBloomData.js # Bloom data hook
│   │   │   └── useAuth.js      # Authentication hook
│   │   ├── utils/
│   │   │   ├── dateUtils.js    # Date utilities
│   │   │   ├── colorScales.js  # Color scale utilities
│   │   │   └── formatters.js   # Data formatting utilities
│   │   ├── App.jsx             # Main application component
│   │   ├── index.jsx           # Application entry point
│   │   └── styles/
│   │       ├── global.css      # Global styles
│   │       └── variables.css   # CSS variables
│   ├── package.json
│   └── vite.config.js          # Vite configuration
├── infrastructure/
│   ├── docker/
│   │   ├── backend.Dockerfile  # Backend Dockerfile
│   │   ├── frontend.Dockerfile # Frontend Dockerfile
│   │   └── docker-compose.yml  # Docker Compose configuration
│   ├── kubernetes/
│   │   ├── backend/            # Backend Kubernetes manifests
│   │   ├── frontend/           # Frontend Kubernetes manifests
│   │   ├── database/           # Database Kubernetes manifests
│   │   └── ingress/            # Ingress Kubernetes manifests
│   └── terraform/              # Infrastructure as Code
├── scripts/
│   ├── setup.sh                # Setup script
│   ├── seed_data.py            # Data seeding script
│   └── deploy.sh               # Deployment script
├── tests/
│   ├── backend/
│   │   ├── data/               # Data processing tests
│   │   ├── models/             # Model tests
│   │   └── api/                # API tests
│   └── frontend/
│       ├── components/         # Component tests
│       └── integration/        # Integration tests
├── docs/
│   ├── architecture.md         # Architecture documentation
│   ├── api.md                  # API documentation
│   ├── models.md               # Model documentation
│   └── deployment.md           # Deployment documentation
├── .gitignore
├── README.md
├── requirements.txt            # Python dependencies
└── pyproject.toml              # Python project configuration
```

## Key Directory Explanations

### Backend

- **data/**: Contains all data processing code
  - **acquisition/**: Interfaces with satellite data sources
  - **preprocessing/**: Data cleaning and preparation
  - **features/**: Feature extraction and engineering

- **models/**: Contains all modeling code
  - **training/**: Model training pipelines
  - **inference/**: Model inference services
  - **evaluation/**: Model evaluation and metrics

- **api/**: FastAPI application
  - **routes/**: API endpoints
  - **schemas/**: Request/response schemas

- **storage/**: Data storage utilities
  - Database connections
  - Raster and vector storage

- **tasks/**: Asynchronous task definitions
  - Background workers
  - Scheduled tasks

### Frontend

- **components/**: React components
  - **Map/**: Map visualization components
  - **Dashboard/**: Dashboard components
  - **Common/**: Shared components

- **services/**: API clients and services

- **hooks/**: React hooks for state management

### Infrastructure

- **docker/**: Docker configurations
- **kubernetes/**: Kubernetes manifests
- **terraform/**: Infrastructure as Code

### Other

- **scripts/**: Utility scripts
- **tests/**: Test suites
- **docs/**: Documentation