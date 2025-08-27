from django.db import models
from cloudinary.models import CloudinaryField
class Tests(models.Model):
    device_id = models.CharField(max_length=255)
    test_type = models.CharField(max_length=100)
    
    # Store uploaded image path
    image = CloudinaryField('image', blank=True, null=True)

    # Keep image_url if you want to store full URL separately
    image_url = models.URLField(blank=True, null=True)

    # Result fields
    result = models.CharField(max_length=20, blank=True)        # positive/negative/invalid
    description = models.TextField(blank=True)                  # GPT reasoning

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.device_id} - {self.test_type} ({self.result})"
