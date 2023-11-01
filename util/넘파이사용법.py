
import numpy as np
#넘파이 생성
a= np.array([0,1,2,3,4,5])

#랜덤 배열 생성
x = np.random.rand(4,4) 

#패딩
x=np.pad(x, pad_width=((0,2),(0,2)),mode="constant",constant_values=0)

#넘파이 추출
a[0] #0번째 값 추출
a[0:3:2] #0에서 3까지, 2칸씩 띄어서 추출
a[0:-1]#마지막 값만 빼고 추출

# 기존의 배열에서 윈도우를 참조하는 방식(참조형)
# 값이 바뀌면 원본도 같이 바뀜
np.arange()

#넘파이 형태 바꾸기
np.expand_dims(a, axis=0) #0번쨰 차원에 차원추가
np.reshape(a,[2,-1]) #2행으로 맞추며, 자동으로 열을 결정(행렬의 개수가 맞아야 한다)