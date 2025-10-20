def filter_spikes( arr ):
    for i in range(1, len(arr) - 1):
        if( (   (arr[i - 1] < arr[i]) and   # bigger than prev
                (arr[i] > arr[i + 1])       # bigger than next
             )
            or
            (   (arr[i - 1] > arr[i]) and   # less than prev
                (arr[i] < arr[i + 1])       # less than next
             )
           ):
            # immediate sign-reversal of the difference from
            # x-1 -> x -> x+1
            prev_dist = abs(arr[i - 1] - arr[i])
            next_dist = abs(arr[i + 1] - arr[i])
            # replace x by the neighboring value that is closest
            # in value
            arr[i] = arr[i - 1] if (prev_dist < next_dist) else arr[i + 1]
            pass;
        pass;
    return arr;


