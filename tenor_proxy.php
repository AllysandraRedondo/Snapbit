<?php
header("Access-Control-Allow-Origin: *");
header("Content-Type: application/json");

$api_key = "AIzaSyAmJB0-hAtiYHm4TLujOs4nYfmD7AlQ9BQ"; // <-- put your actual API key here
$q = urlencode($_GET['q'] ?? '');
$url = "https://tenor.googleapis.com/v2/search?q=$q&key=$api_key&client_key=Snapbits&media_filter=sticker&limit=12";

echo file_get_contents($url);
?>
