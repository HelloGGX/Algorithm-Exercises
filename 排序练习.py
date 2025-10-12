def partition(arr, low, high):
    mid = (low + high)//2
    dia = [(arr[low], low), (arr[mid], mid), (arr[high], high)]
    dia.sort()
    pivot_idx = dia[1][1]
    pivot = arr[pivot_idx]
    
    arr[pivot_idx], arr[high] = arr[high], arr[pivot_idx]
    
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i+=1
            arr[i], arr[j] = arr[j], arr[i]
        
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    
    return i + 1


# 插入排序
def insert_sort(arr):
    for i in range(1, len(arr)):
        j = i-1
        key = arr[i]
        while j >= 0 and arr[j] > key:
            arr[j+1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr