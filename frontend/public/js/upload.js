/**
 * BloomTrack Upload Module
 * Gestisce le funzionalità di upload dei file con validazione
 */
class FileUploader {
    constructor(apiBaseUrl = '/api') {
        this.apiBaseUrl = apiBaseUrl;
        this.uploadEndpoint = `${this.apiBaseUrl}/upload/upload`;
        this.listFilesEndpoint = `${this.apiBaseUrl}/upload/files`;
        this.deleteFileEndpoint = `${this.apiBaseUrl}/upload/files`;
        this.previewEndpoint = `${this.apiBaseUrl}/upload/preview`;
    }

    /**
     * Carica un file al server
     * @param {File} file - Il file da caricare
     * @param {string} description - Descrizione opzionale del file
     * @param {boolean} skipValidation - Se saltare la validazione del file
     * @returns {Promise} - Promise con la risposta del server
     */
    async uploadFile(file, description = '', skipValidation = false) {
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            if (description) {
                formData.append('description', description);
            }
            
            formData.append('skip_validation', skipValidation);

            const response = await fetch(this.uploadEndpoint, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Errore durante l\'upload');
            }

            return await response.json();
        } catch (error) {
            console.error('Errore di upload:', error);
            throw error;
        }
    }

    /**
     * Ottiene la lista dei file caricati
     * @param {string} fileType - Filtra per tipo di file
     * @param {boolean} validOnly - Mostra solo file validi
     * @returns {Promise} - Promise con la lista dei file
     */
    async getFilesList(fileType = '', validOnly = false) {
        try {
            let endpoint = this.listFilesEndpoint;
            const params = new URLSearchParams();
            
            if (fileType) {
                params.append('file_type', fileType);
            }
            
            if (validOnly) {
                params.append('valid_only', 'true');
            }
            
            const queryString = params.toString();
            if (queryString) {
                endpoint += `?${queryString}`;
            }
            
            const response = await fetch(endpoint);
            
            if (!response.ok) {
                throw new Error('Errore nel recupero della lista dei file');
            }

            return await response.json();
        } catch (error) {
            console.error('Errore nel recupero dei file:', error);
            throw error;
        }
    }

    /**
     * Elimina un file
     * @param {string} filename - Il nome del file da eliminare
     * @returns {Promise} - Promise con la risposta del server
     */
    async deleteFile(filename) {
        try {
            const response = await fetch(`${this.deleteFileEndpoint}/${filename}`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                throw new Error('Errore nell\'eliminazione del file');
            }

            return await response.json();
        } catch (error) {
            console.error('Errore nell\'eliminazione del file:', error);
            throw error;
        }
    }
    
    /**
     * Ottiene l'anteprima di un file
     * @param {string} filename - Il nome del file da visualizzare
     * @returns {Promise} - Promise con i dati dell'anteprima
     */
    async getFilePreview(filename) {
        try {
            const response = await fetch(`${this.previewEndpoint}/${filename}`);
            
            if (!response.ok) {
                throw new Error('Errore nel recupero dell\'anteprima');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Errore nel recupero dell\'anteprima:', error);
            throw error;
        }
    }
}

// Inizializzazione quando il documento è pronto
document.addEventListener('DOMContentLoaded', () => {
    const uploader = new FileUploader();
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const descriptionInput = document.getElementById('description-input');
    const filesList = document.getElementById('files-list');
    const uploadStatus = document.getElementById('upload-status');
    const loadingIndicator = document.getElementById('loading-indicator');

    // Gestione dell'upload
    if (uploadForm) {
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            if (!fileInput.files || fileInput.files.length === 0) {
                showStatus('Seleziona un file da caricare', 'error');
                return;
            }

            const file = fileInput.files[0];
            const description = descriptionInput ? descriptionInput.value : '';
            
            try {
                showLoading(true);
                const result = await uploader.uploadFile(file, description);
                showStatus(`File caricato con successo: ${result.filename}`, 'success');
                
                // Aggiorna la lista dei file
                loadFilesList();
                
                // Reset del form
                uploadForm.reset();
            } catch (error) {
                showStatus(`Errore: ${error.message}`, 'error');
            } finally {
                showLoading(false);
            }
        });
    }

    // Carica la lista dei file all'avvio
    loadFilesList();

    // Funzione per caricare la lista dei file
    async function loadFilesList() {
        if (!filesList) return;
        
        try {
            showLoading(true);
            const data = await uploader.getFilesList();
            
            // Pulisci la lista
            filesList.innerHTML = '';
            
            if (data.files.length === 0) {
                filesList.innerHTML = '<p>Nessun file caricato</p>';
                return;
            }
            
            // Crea la lista dei file
            data.files.forEach(file => {
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                
                // Formatta la dimensione del file
                const sizeStr = formatFileSize(file.size);
                
                fileItem.innerHTML = `
                    <div class="file-info">
                        <span class="file-name">${file.filename}</span>
                        <span class="file-size">${sizeStr}</span>
                        <span class="file-type">${file.content_type}</span>
                    </div>
                    <div class="file-actions">
                        <button class="delete-btn" data-filename="${file.filename}">Elimina</button>
                    </div>
                `;
                
                filesList.appendChild(fileItem);
                
                // Aggiungi event listener per il pulsante di eliminazione
                const deleteBtn = fileItem.querySelector('.delete-btn');
                deleteBtn.addEventListener('click', async () => {
                    if (confirm('Sei sicuro di voler eliminare questo file?')) {
                        try {
                            showLoading(true);
                            await uploader.deleteFile(file.filename);
                            showStatus('File eliminato con successo', 'success');
                            loadFilesList();
                        } catch (error) {
                            showStatus(`Errore nell'eliminazione: ${error.message}`, 'error');
                        } finally {
                            showLoading(false);
                        }
                    }
                });
            });
        } catch (error) {
            filesList.innerHTML = `<p class="error">Errore nel caricamento della lista: ${error.message}</p>`;
        } finally {
            showLoading(false);
        }
    }

    // Funzione per mostrare lo stato
    function showStatus(message, type = 'info') {
        if (!uploadStatus) return;
        
        uploadStatus.textContent = message;
        uploadStatus.className = `status ${type}`;
        uploadStatus.style.display = 'block';
        
        // Nascondi dopo 5 secondi
        setTimeout(() => {
            uploadStatus.style.display = 'none';
        }, 5000);
    }

    // Funzione per mostrare/nascondere l'indicatore di caricamento
    function showLoading(show) {
        if (!loadingIndicator) return;
        loadingIndicator.style.display = show ? 'block' : 'none';
    }

    // Funzione per formattare la dimensione del file
    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        else if (bytes < 1073741824) return (bytes / 1048576).toFixed(1) + ' MB';
        else return (bytes / 1073741824).toFixed(1) + ' GB';
    }
});