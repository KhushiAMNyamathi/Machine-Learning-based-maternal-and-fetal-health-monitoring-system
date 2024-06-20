$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
	
	$('#sol0').hide();
	$('#sol1').hide();
		
	$('#sol2').hide();
	$('#sol3').hide();
        

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
		$('#sol0').hide();
		$('#sol1').hide();
		
		$('#sol2').hide();
		$('#sol3').hide();
        
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
				console.log(data);
                $('.loader').hide();
                $('#result').fadeIn(600);
				
				
				if(data=="0")
				{
					
					msg="Corn common Rust";
					
					$('#result').text(' Result:  ' + msg);
					$('#sol0').show();
                
				}
						
				else if(data=="1")
				{
					
					msg="Potato early blight";
					
					$('#result').text(' Result:  ' + msg);
					$('#sol1').show();
                
				}
				else if(data=="2")
				{
					
					msg="Tomato Bacterial Spot";
					
					$('#result').text(' Result:  ' + msg);
					$('#sol1').show();
                
				}
				else if(data=="3")
				{
					
					msg="Potato early blight";
					
					$('#result').text(' Result:  ' + msg);
					$('#sol1').show();
                
				}
				else if( data=="4"){
				
				msg="Healthy tomato";
				$('#result').text(' Result:  ' + msg);
				}
                console.log('Success!');
				
				
            },
        });
    });

});
