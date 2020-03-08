<?php

function eratosthenes($limit) {
    for ($i=2; $i<=$limit; $i++) {
        $nums[$i]=true;
    }
    for ($j=2; $j < $limit; $j++) {
        $key = $j;
        $val = $nums[$key];
        if (!is_array($val)) { // not visited yet, $key must be prime
            for ($i=2*$key; $i <= $limit; $i+=$key) { // iterate over all of its multiplications
                if ($nums[$i] == true) {
                    $nums[$i] = [
                        $key => 1,
                    ];

                    $mult = $i/$key;

                    if(!is_array($nums[$mult])) {
                        $nums[$i][$mult] += 1;
                    } else {
                        foreach($nums[$mult] as $k => $v) {
                            $nums[$i][$k] += $v;
                        }
                    }
                }
            }
        }

    }

    return $nums;
}

$limit = $argv[1];

if ($limit < 2 || $limit > 1000000) {
    echo "Limit must be between 2 and 1000000\n";
    exit(1);
}

foreach(eratosthenes($limit) as $k => $v) {
    $res = $k;
    if (is_array($v)) {
        ksort($v);
        foreach($v as $j => $d) {
            $res = $res." ".$j."^".$d;
        }
    }

    echo strval($res)."\n";
}

?>

