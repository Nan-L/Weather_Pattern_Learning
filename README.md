In this assignment, the idea here is to take the images as input, match them up with known weather conditions, and try to learn to 
produce a description of the weather.

The weather data has a column “weather” that we can use as a target: it's not there for every hour, but there's enough to train with.

Doing this will require two data sets: webcam images, and known weather observations.

Kat Kam has been taking lovely images of English Bay since 1996. They have given us permission to use some of their archived images 
for this project.

I have collected as many archived images as I can, so we can use them without all hitting their server. The original images are 
1280×960, which is probably too large to be practical. I have a set scaled to 256×192, which should be enough for us to work with. 
You can download the Kat Kam images.

Since the images are hourly (during daylight hours), the GHCN weather data we have been working with isn't specific enough: it has 
only daily readings. Fortunately, the Canadian government posts historical weather data that has hourly readings. I have collected 
those as well, from the Vancouver Airport weather station, which has the most complete data. You can download the weather data for 
the relevant months.
