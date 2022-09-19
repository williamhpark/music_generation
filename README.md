## Music Generation Project

### Summary

Music has always been a huge part of my life. I've played the guitar since I was 11 year old, trumpet throughout all of high school, and just recently I started exploring music production.

My recent project of learning to produce music using the Ableton DAW has been a lot of fun, but also very frustrating, especially when I'm trying to think of melodies to base the songs off of. This made me consider how machine learning can be used to generate melodies, which I could maybe use for inspiration when producing, and was thus the source of inspiration for this project.

### Live URL

### Technologies

- TensorFlow
- Flask
- Heroku

### Next Steps

- Improve the frontend UI, possibly with the help of a frontend library (e.g. React)
- Experiment with tuning the parameters (i.e. learning rate, epochs, loss weights, etc) for lower total loss
- Deploy and train the model on a cloud service (e.g. Amazon SageMaker)
- Allow the user to customize the melody, e.g. length, speed
- Allow the user to generate melodies based on a selected genre (this would require the use of a different dataset where genre metadata is available)
- Explore usig GANs for music generation

### References

- https://www.tensorflow.org/tutorials/audio/music_generation
