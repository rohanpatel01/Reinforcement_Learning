
from datetime import datetime


# x = datetime.date(datetime.now())
# t = datetime.time(datetime.now()).

formatted_datetime = datetime.now().strftime("Date:%Y-%m-%d_Time%H-%M-%S")
print(formatted_datetime)

# print(x)
# print(t)