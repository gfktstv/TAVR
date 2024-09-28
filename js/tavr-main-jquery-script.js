// Connect jquery for radio buttons actions
var script = document.createElement('script');
script.src = 'https://code.jquery.com/jquery-3.7.1.min.js';
document.getElementsByTagName('head')[0].appendChild(script);

// Quick analytics radio buttons actions
$(document).ready(function() {
    $("input[name$='quick-analytics-choice']").click(function() {
        var test = $(this).val();

        $('div.desc').css('max-height', '0px');
        setTimeout(function() {
            $('div.desc').hide();
            $('#' + test + 'ContainerQckAnltcs').show();
            $('#' + test + 'ContainerQckAnltcs').css('max-height', '300px')
        }, 550);
    });
  });