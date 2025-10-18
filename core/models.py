import uuid  
from django.db import models
from django.contrib.auth.models import User

class AnalysisRun(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    filename = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

class AnalysisHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    reference = models.CharField(max_length=255)
    finding = models.TextField()
    reasons = models.TextField()
    root_cause = models.TextField()
    score = models.IntegerField(null=True, blank=True)  # ← allow NULL in DB
    status = models.CharField(max_length=50)
    comment = models.TextField()
    measures = models.TextField()
    ca_score   = models.IntegerField(null=True, blank=True)        # ← allow NULL in DB
    ca_status  = models.CharField(max_length=32, null=True, blank=True)   # ← make nullable
    ca_comment = models.TextField(null=True, blank=True)                  # ← make nullable
    created_at = models.DateTimeField(auto_now_add=True)
    run = models.ForeignKey(
        AnalysisRun, related_name="items",
        on_delete=models.CASCADE, null=True, blank=True
    )

    def __str__(self):
        return f"{self.reference} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
