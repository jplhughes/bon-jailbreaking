$(document).ready(function() {

  var editor = CodeMirror.fromTextArea(document.getElementById("bibtex"), {
      lineNumbers: false,
      lineWrapping: true,
      readOnly:true
  });
  $(function () {
      $('[data-toggle="tooltip"]').tooltip()
  });

  var currentDataset = 'text';
  var datasetUrls = {
    'text': 'assets/data/text_jailbreaks.csv',
    'vision': 'assets/data/vision_jailbreaks.csv', 
    'audio': 'assets/data/audio_jailbreaks.csv'
  };

  function loadData(dataset) {
    // Clear existing options first
    $('#model-filter').html('<option value="">All Models</option>');
    $('#behavior-filter').html('<option value="">All Behaviors</option>');
    
    $.ajax({
      type: "GET", 
      url: datasetUrls[dataset],
      dataType: "text",
      success: function(data) {
        var csvData = Papa.parse(data, {header: true}).data;
        var currentIndex = 0;

        function displayEntry(entry, filteredLength) {
          $('#entries-container').empty();
            
          // Common header
          let html = `
            <div class="entry">
              <h4>Model: ${entry.model}</h4>
              <p><strong>Request:</strong> ${entry.direct_request}</p>`;

          // Dataset specific content
          if (dataset === 'text') {
            html += `
              <p><strong>Augmented Request:</strong> ${entry.augmented_file}</p>`;
          } else if (dataset === 'audio') {
            html += `
              <p><strong>Sampled augmentation values:</strong></p>
              <div class="row">
                <div class="col-md-4" style="padding-left: 40px;">
                  <p>🏃🏽‍♀️ Speed: ${parseFloat(entry.speed).toFixed(2)}x</p>
                </div>
                <div class="col-md-4" style="padding-left: 40px;">
                  <p>⚡ Pitch: ${(parseFloat(entry.pitch) / 100).toFixed(2)} semitones</p>
                </div>
                <div class="col-md-4" style="padding-left: 40px;">
                  <p>🗣️ Speech: ${parseFloat(entry.speech).toFixed(2)}dB</p>
                </div>
                <div class="col-md-4" style="padding-left: 40px;">
                  <p>🚗 Noise: ${parseFloat(entry.noise).toFixed(2)}dB</p>
                </div>
                <div class="col-md-4" style="padding-left: 40px;">
                  <p>📢 Volume: ${parseFloat(entry.volume).toFixed(3)}x</p>
                </div>
                <div class="col-md-4" style="padding-left: 40px;">
                  <p>🎹 Music: ${parseFloat(entry.music).toFixed(2)}dB</p>
                </div>
              </div>
              <div class="row">
                <div class="col-md-6">
                  <p><strong>Vocalized Request:</strong></p>
                  <audio controls>
                    <source src="assets/${entry.audio_file}" type="audio/mpeg">
                    Your browser does not support the audio element.
                  </audio>
                </div>
                <div class="col-md-6">
                  <p><strong>Augmented Request:</strong></p>
                  <audio controls>
                    <source src="assets/${entry.augmented_file}" type="audio/mpeg">
                    Your browser does not support the audio element.
                  </audio>
                </div>
              </div>`;
          } else if (dataset === 'vision') {
            const dimensionsId = `dimensions-${entry.augmented_file.replace(/[^a-zA-Z0-9]/g, '-')}`;
            html += `
              <p><strong>Sampled augmentation values:</strong></p>
              <div class="row">
                <div class="col-md-4" style="padding-left: 40px;">
                  <p>📍 Position: ${entry.position}</p>
                </div>
                <div class="col-md-4" style="padding-left: 40px;">
                  <p>🔤 Font: ${entry.font}</p>
                </div>
                <div class="col-md-4" style="padding-left: 40px;">
                  <p>📏 Font Scale: ${parseFloat(entry.font_scale).toFixed(2)}</p>
                </div>
                <div class="col-md-4" style="padding-left: 40px;">
                  <p>🎨 Color: ${entry.color}</p>
                </div>
                <div class="col-md-4" style="padding-left: 40px;">
                  <p>✏️ Thickness: ${entry.thickness}</p>
                </div>
                <div class="col-md-4" style="padding-left: 40px;">
                  <p>🖼️ Dimensions: <span id="${dimensionsId}">Loading...</span></p>
                </div>
              </div>
              <p><strong>Augmented Image:</strong></p>
              <img src="assets/${entry.augmented_file}" 
                   style="max-width: 100%; height: auto;"
                   onload="(function(img) {
                     const dimensionsEl = document.getElementById('${dimensionsId}');
                     if (dimensionsEl) {
                       dimensionsEl.textContent = '(' + img.naturalWidth + ', ' + img.naturalHeight + ')';
                     }
                   })(this)">
              <div style="margin-top: 20px;">`;
          }

          // Common response section
          html += `
            <p><strong>Model Response:</strong></p>
            <div style="background-color: white; padding: 10px; border: 1px solid #ddd; border-radius: 10px;">
              <p>${entry.response.replace(/\n/g, '<br>').split(' ').slice(0,50).join(' ')} [TRUNCATED]</p>
            </div>
            <p style="font-size: 0.9em; color: #888;">Note: responses are limited to 50 words to show evidence of jailbroken behavior without revealing all harmful content.</p>`;

          // Audio-specific speech output
          if (dataset === 'audio' && entry.speech_out_file) {
            html += `
              <p><strong>Speech Output File:</strong></p>
              <audio controls>
                <source src="assets/${entry.speech_out_file}" type="audio/mpeg">
                Your browser does not support the audio element.
              </audio>`;
          }

          html += `</div>`;

          $('#entries-container').append(html);
          $('#entry-info').text(`Entry ${currentIndex + 1} of ${filteredLength}`);
          $('#entry-seek').attr('max', filteredLength);
        }

        function updateEntry() {
          var selectedModel = $('#model-filter').val();
          var selectedBehavior = $('#behavior-filter').val();
          var filteredEntries = csvData.filter(entry => {
            return (!selectedModel || entry.model === selectedModel) && (!selectedBehavior || entry.behavior_id === selectedBehavior);
          });
          if (filteredEntries.length > 0) {
            displayEntry(filteredEntries[currentIndex], filteredEntries.length);
            
            // Enable/disable navigation buttons based on current index
            $('#prev-entry').prop('disabled', currentIndex === 0);
            $('#next-entry').prop('disabled', currentIndex >= filteredEntries.length - 1);
          } else {
            $('#entries-container').empty();
            $('#entry-info').text('No entries found');
            $('#prev-entry').prop('disabled', true);
            $('#next-entry').prop('disabled', true);
          }
        }

        // Function to update model filter
        function updateModelFilter(selectedBehavior) {
          // Get current selected model to preserve it if possible
          const currentModel = $('#model-filter').val();
          
          var filteredModels = selectedBehavior ?
            [...new Set(csvData.filter(entry => entry.behavior_id === selectedBehavior).map(entry => entry.model))] :
            [...new Set(csvData.map(entry => entry.model))];
          
          filteredModels = filteredModels.filter(model => model);
          filteredModels.sort();
          
          $('#model-filter').html('<option value="">All Models</option>');
          filteredModels.forEach(model => {
            $('#model-filter').append(`<option value="${model}">${model}</option>`);
          });

          // Restore previous selection if it's still valid
          if (currentModel && filteredModels.includes(currentModel)) {
            $('#model-filter').val(currentModel);
          }
        }

        // Function to update behavior filter
        function updateBehaviorFilter(selectedModel) {
          // Get current selected behavior to preserve it if possible
          const currentBehavior = $('#behavior-filter').val();
          
          var filteredBehaviors = selectedModel ? 
            [...new Set(csvData.filter(entry => entry.model === selectedModel).map(entry => entry.behavior_id))] :
            [...new Set(csvData.map(entry => entry.behavior_id))];
          
          filteredBehaviors = filteredBehaviors.filter(behavior => behavior);
          filteredBehaviors.sort();
          
          $('#behavior-filter').html('<option value="">All Behaviors</option>');
          filteredBehaviors.forEach(behavior => {
            $('#behavior-filter').append(`<option value="${behavior}">${behavior}</option>`);
          });

          // Restore previous selection if it's still valid
          if (currentBehavior && filteredBehaviors.includes(currentBehavior)) {
            $('#behavior-filter').val(currentBehavior);
          }
        }

        // Initial filter population
        updateModelFilter('');
        updateBehaviorFilter('');

        // Event handlers
        $('#model-filter').change(() => {
          var selectedModel = $('#model-filter').val();
          // Only update behavior filter, don't reset model filter
          updateBehaviorFilter(selectedModel);
          currentIndex = 0;
          updateEntry();
        });

        $('#behavior-filter').change(() => {
          var selectedBehavior = $('#behavior-filter').val();
          // Only update model filter, don't reset behavior filter
          updateModelFilter(selectedBehavior);
          currentIndex = 0;
          updateEntry();
        });

        $('#next-entry').click(() => {
          if (currentIndex < csvData.length - 1) {
            currentIndex++;
            updateEntry();
          }
        });

        $('#prev-entry').click(() => {
          if (currentIndex > 0) {
            currentIndex--;
            updateEntry();
          }
        });

        $('#seek-entry').click(() => {
          var seekIndex = parseInt($('#entry-seek').val(), 10) - 1;
          var filteredEntries = csvData.filter(entry => {
            var selectedModel = $('#model-filter').val();
            var selectedBehavior = $('#behavior-filter').val();
            return (!selectedModel || entry.model === selectedModel) && (!selectedBehavior || entry.behavior_id === selectedBehavior);
          });
          if (seekIndex >= 0 && seekIndex < filteredEntries.length) {
            currentIndex = seekIndex;
            updateEntry();
          }
        });

        updateEntry();
      }
    });
  }

  // Tab click handlers
  $('.nav-tabs a').click(function(e) {
    e.preventDefault();
    $(this).tab('show');
    currentDataset = $(this).attr('href').substring(1);
    loadData(currentDataset);
  });

  // Initial load
  loadData(currentDataset);
});