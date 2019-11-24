var express = require('express')
var bodyParser = require('body-parser');
var app = express();

app.use(bodyParser.urlencoded({extended: false}));
app.use(bodyParser.json());
app.use(express.static('www'));

app.post('handle', function(request, response){
var query1 = request.body.var1;
var query2 = request.body.var2; 
console.log("test");
});


var server = app.listen(8000, function () {

    var host = server.address().address
    var port = server.address().port

    console.log('Express app listening at http://%s:%s', host, port)

})
