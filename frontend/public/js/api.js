/**
 * api.js - API client for BloomTrack
 * 
 * This module handles communication with the BloomTrack API endpoints.
 */

class BloomTrackAPI {
    constructor(baseUrl = '/api') {
        this.baseUrl = baseUrl;
    }

    /**
     * Get a prediction for a specific location and date
     * @param {number} latitude - Latitude
     * @param {number} longitude - Longitude
     * @param {string} date - Date in ISO format
     * @param {boolean} includeForecast - Whether to include forecast data
     * @param {number} forecastDays - Number of days to forecast
     * @returns {Promise<object>} - Prediction data
     */
    async getPredictionForLocation(latitude, longitude, date, includeForecast = false, forecastDays = 30) {
        try {
            const response = await fetch(`${this.baseUrl}/predict/location`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    latitude,
                    longitude,
                    date,
                    include_forecast: includeForecast,
                    forecast_days: forecastDays
                })
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.status} ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error fetching prediction:', error);
            throw error;
        }
    }

    /**
     * Get a prediction for a bounding box
     * @param {number} minLat - Minimum latitude
     * @param {number} minLon - Minimum longitude
     * @param {number} maxLat - Maximum latitude
     * @param {number} maxLon - Maximum longitude
     * @param {string} date - Date in ISO format
     * @param {number} resolution - Grid resolution in degrees
     * @returns {Promise<object>} - Prediction data with raster and tile URLs
     */
    async getPredictionForBoundingBox(minLat, minLon, maxLat, maxLon, date, resolution = 0.01) {
        try {
            const response = await fetch(`${this.baseUrl}/predict/bbox`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    min_lat: minLat,
                    min_lon: minLon,
                    max_lat: maxLat,
                    max_lon: maxLon,
                    date,
                    resolution
                })
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.status} ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error fetching bounding box prediction:', error);
            throw error;
        }
    }

    /**
     * Get information about the loaded models
     * @returns {Promise<object>} - Model information
     */
    async getModelInfo() {
        try {
            const response = await fetch(`${this.baseUrl}/model/info`);

            if (!response.ok) {
                throw new Error(`API error: ${response.status} ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error fetching model info:', error);
            throw error;
        }
    }

    /**
     * Check the health of the API
     * @returns {Promise<object>} - Health status
     */
    async checkHealth() {
        try {
            const response = await fetch(`${this.baseUrl}/health`);

            if (!response.ok) {
                throw new Error(`API error: ${response.status} ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error checking API health:', error);
            throw error;
        }
    }

    /**
     * Get a prediction for the current map view
     * @param {L.Map} map - Leaflet map instance
     * @param {string} date - Date in ISO format
     * @param {number} resolution - Grid resolution in degrees
     * @returns {Promise<object>} - Prediction data with raster and tile URLs
     */
    async getPredictionForMapView(map, date, resolution = 0.01) {
        const bounds = map.getBounds();
        const southWest = bounds.getSouthWest();
        const northEast = bounds.getNorthEast();

        return this.getPredictionForBoundingBox(
            southWest.lat,
            southWest.lng,
            northEast.lat,
            northEast.lng,
            date,
            resolution
        );
    }
}

// Export the API client
window.BloomTrackAPI = BloomTrackAPI;