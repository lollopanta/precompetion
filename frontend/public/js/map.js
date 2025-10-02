/**
 * map.js - Map visualization for BloomTrack
 * 
 * This module handles the Leaflet map initialization, layer management,
 * and interaction with the map for the BloomTrack application.
 */

class BloomMap {
    constructor(mapElementId) {
        this.mapElementId = mapElementId;
        this.map = null;
        this.baseLayers = {};
        this.overlayLayers = {};
        this.markers = [];
        this.currentBloomLayer = null;
        this.legend = null;
        this.initialize();
    }

    /**
     * Initialize the map and base layers
     */
    initialize() {
        // Create the map
        this.map = L.map(this.mapElementId, {
            center: [20, 0],
            zoom: 2,
            minZoom: 2,
            maxZoom: 18
        });

        // Add base layers
        this.baseLayers.openStreetMap = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(this.map);

        this.baseLayers.satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
        });

        // Add legend
        this.addLegend();

        // Set up event handlers
        this.setupEventHandlers();
    }

    /**
     * Set up event handlers for map interactions
     */
    setupEventHandlers() {
        // Handle map clicks to set coordinates in the form
        this.map.on('click', (e) => {
            const lat = e.latlng.lat.toFixed(6);
            const lng = e.latlng.lng.toFixed(6);
            
            // Update form inputs
            document.getElementById('latitude').value = lat;
            document.getElementById('longitude').value = lng;
            
            // Add a marker at the clicked location
            this.setMarker(lat, lng);
        });

        // Handle layer toggle checkboxes
        document.getElementById('bloom-layer').addEventListener('change', (e) => {
            if (e.target.checked) {
                if (this.currentBloomLayer) {
                    this.currentBloomLayer.addTo(this.map);
                }
            } else {
                if (this.currentBloomLayer) {
                    this.map.removeLayer(this.currentBloomLayer);
                }
            }
        });

        document.getElementById('satellite-layer').addEventListener('change', (e) => {
            if (e.target.checked) {
                this.map.removeLayer(this.baseLayers.openStreetMap);
                this.baseLayers.satellite.addTo(this.map);
            } else {
                this.map.removeLayer(this.baseLayers.satellite);
                this.baseLayers.openStreetMap.addTo(this.map);
            }
        });
    }

    /**
     * Add a marker to the map at the specified coordinates
     * @param {number} lat - Latitude
     * @param {number} lng - Longitude
     */
    setMarker(lat, lng) {
        // Clear existing markers
        this.clearMarkers();
        
        // Add new marker
        const marker = L.marker([lat, lng]).addTo(this.map);
        this.markers.push(marker);
        
        // Center map on marker
        this.map.setView([lat, lng], Math.max(this.map.getZoom(), 10));
        
        return marker;
    }

    /**
     * Clear all markers from the map
     */
    clearMarkers() {
        this.markers.forEach(marker => this.map.removeLayer(marker));
        this.markers = [];
    }

    /**
     * Add a bloom prediction layer to the map
     * @param {string} url - URL to the tile layer
     * @param {object} options - Layer options
     */
    addBloomLayer(url, options = {}) {
        // Remove existing bloom layer if any
        if (this.currentBloomLayer) {
            this.map.removeLayer(this.currentBloomLayer);
        }
        
        // Create new layer with default options
        const defaultOptions = {
            opacity: 0.7,
            attribution: 'BloomTrack Prediction'
        };
        
        const layerOptions = { ...defaultOptions, ...options };
        
        // Create and add the layer
        this.currentBloomLayer = L.tileLayer(url, layerOptions);
        
        // Only add to map if the bloom layer checkbox is checked
        if (document.getElementById('bloom-layer').checked) {
            this.currentBloomLayer.addTo(this.map);
        }
        
        return this.currentBloomLayer;
    }

    /**
     * Add a legend to the map
     */
    addLegend() {
        if (this.legend) {
            this.map.removeControl(this.legend);
        }
        
        this.legend = L.control({ position: 'bottomright' });
        
        this.legend.onAdd = () => {
            const div = L.DomUtil.create('div', 'map-legend');
            div.innerHTML = `
                <h6>Bloom Probability</h6>
                <div class="legend-gradient"></div>
                <div class="legend-labels">
                    <span>0%</span>
                    <span>50%</span>
                    <span>100%</span>
                </div>
            `;
            return div;
        };
        
        this.legend.addTo(this.map);
    }

    /**
     * Show a popup with bloom prediction information
     * @param {number} lat - Latitude
     * @param {number} lng - Longitude
     * @param {object} data - Prediction data
     */
    showPredictionPopup(lat, lng, data) {
        const marker = this.setMarker(lat, lng);
        
        const probabilityPercent = Math.round(data.bloom_probability * 100);
        const intensityValue = data.bloom_intensity ? Math.round(data.bloom_intensity * 10) / 10 : 0;
        
        const popupContent = `
            <div class="popup-content">
                <h5>Bloom Prediction</h5>
                <p><strong>Status:</strong> ${data.is_blooming ? 'Blooming' : 'Not Blooming'}</p>
                <p><strong>Probability:</strong> ${probabilityPercent}%</p>
                <div class="progress">
                    <div class="progress-bar" role="progressbar" style="width: ${probabilityPercent}%" 
                        aria-valuenow="${probabilityPercent}" aria-valuemin="0" aria-valuemax="100"></div>
                </div>
                <p><strong>Intensity:</strong> ${intensityValue}/10</p>
                <div class="progress">
                    <div class="progress-bar bg-success" role="progressbar" style="width: ${intensityValue * 10}%" 
                        aria-valuenow="${intensityValue}" aria-valuemin="0" aria-valuemax="10"></div>
                </div>
                <p><strong>Date:</strong> ${new Date(data.prediction_date).toLocaleDateString()}</p>
            </div>
        `;
        
        marker.bindPopup(popupContent).openPopup();
    }

    /**
     * Fit the map to a bounding box
     * @param {number} minLat - Minimum latitude
     * @param {number} minLng - Minimum longitude
     * @param {number} maxLat - Maximum latitude
     * @param {number} maxLng - Maximum longitude
     */
    fitBounds(minLat, minLng, maxLat, maxLng) {
        const bounds = L.latLngBounds(
            L.latLng(minLat, minLng),
            L.latLng(maxLat, maxLng)
        );
        
        this.map.fitBounds(bounds);
    }

    /**
     * Show a loading indicator on the map
     * @param {boolean} show - Whether to show or hide the indicator
     */
    showLoading(show = true) {
        const existingIndicator = document.querySelector('.loading-indicator');
        
        if (show) {
            if (!existingIndicator) {
                const loadingDiv = document.createElement('div');
                loadingDiv.className = 'loading-indicator';
                loadingDiv.innerHTML = `
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-2">Loading prediction data...</p>
                `;
                
                document.querySelector('.map-container').appendChild(loadingDiv);
            }
        } else if (existingIndicator) {
            existingIndicator.remove();
        }
    }
}

// Export the BloomMap class
window.BloomMap = BloomMap;