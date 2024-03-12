FROM public.ecr.aws/lambda/python:3.11

COPY . .


ENV TRANSFORMERS_CACHE "/tmp/transformers_cache"

# Create a writable directory
RUN mkdir -p /tmp/transformers_cache && chmod -R 777 /tmp/transformers_cache

# Create a directory for data
#RUN mkdir '/tmp/data/'
COPY Big-Five-Personality-Traits-Detection '/usr/share/'


# Copy requirements.txt and install the required packages
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy app.py
COPY app.py ${LAMBDA_TASK_ROOT}

# Set the CMD for the Lambda function
CMD [ "app.handler" ]
