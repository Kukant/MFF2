<?php

/*
 * Initial parameters, which are loaded from command line.
 * Each static member variable is annotated by doc-comments, so it can be automatically parsed
 * and a help (usage) can be automatically printed.
 * Possible annotations:
 * @required - if present, the argument is mandatory
 * @string, @int, @bool - define a data type of the argument value (bool has no value, argument presence ~ true, false should be default).
 * @short(str) - shorthand argument string (e.g., $base has @short(b), which means the argument can be written as --base or -b).
 * 		Note that aggregation (-b -l written as -bl) of short arguments is not required
 */
class Args
{
	/**
	 * The URL base for generating relative URLs.
	 * The base must match complete prefix of an URL to be applied.
	 * @required @string @short(b)
	 */
	public static $base = null;

	/**
	 * If present, <a> elements are left unchanged.
	 * @bool
	 */
	public static $no_links = false;

	/**
	 * If present, <img> elements are left unchanged.
	 * @bool
	 */
	public static $no_imgs = false;

	/**
	 * If present, all other elements except for <a> and <img> (i.e., <script>, <form>, <frame>, ...) are left unchanged.
	 * @bool
	 */
	public static $no_others = false;


	/**
	 * Minimal length, that an absolute URL must have to be replaced. Shorther URLs will remain absolute.
	 * @int @short(l)
	 */
	public static $length = null;



	/*
	 * Parsing Methods
	 */


	/**
	 * Load the arguments from an array (e.g., the $argv may be passed down right away).
	 * First value of the args array is expected to be path to this script.
	 * @return remaining unprocessed arguments from the $args array.
	 */
	public static function load(array $args)
	{
		array_shift($args); // remove the path
        $props = self::parseProperties();
        $reflector = new ReflectionClass('Args');
        $remove = Array();
        for ($i=0; $i < count($args); $i++) {
            $arg = self::getArg($args[$i], $props);

            if ($arg == null) {
                if (substr( $args[$i], 0, 1 ) === "-") {
                    self::printHelp($props);
                    exit(12);
                }
                break;
            }

            if ($arg["type"] == "bool") {
                $value = true;
            } else {
                if ($i + 1 >= count($args)) {
                    throw new Exception("Not enough arguments.");
                }

                if ($arg["type"] == "int") {
                    $value = (int)$args[$i + 1];
                } else {
                    $value = $args[$i + 1];
                }
                array_unshift($remove, $i);
                $i++;
            }
           // echo "Setting ".$arg["name"]." to ".$value."\n";

            $reflector->getProperty($arg["name"])->setValue($value);
            array_unshift($remove, $i);

        }

        $remaining = Array();
        foreach ($args as $k => $arg) {
            if (!in_array($k, $remove))
                array_push($remaining, $arg);
        }

		return $remaining;
	}

    static function printHelp($props) {
	    foreach($props as $k => $v) {
	        echo $v."\n";
        }
    }

	static function parseProperties() {
        $reflector = new ReflectionClass('Args');
        $properties = $reflector->getProperties();

        // {}
        $pp = Array();

        $pattern = "#([^\*\/]+)#";
        $short_pattern = "#short\(([a-zA-Z])\)#";

        foreach($properties as  $key => $p) {
            $comment = $p->getDocComment();
            preg_match_all($pattern, $comment, $matches);
            $parsed_comment = array_filter($matches[0], function($value) {
                return trim($value) !== '';
            });

            $last_line = end($parsed_comment);
            preg_match($short_pattern, $last_line, $short_match);
            $short = count($short_match) > 0 ? end($short_match) : null;
            $pp[$p->name] = [
                "name" => $p->name,
                "message" => join("\n", $parsed_comment),
                "long" => "--".$p->name,
                "short" => "-".$short,
                "required" => strpos($last_line, "@required"),
                "type" => strpos($last_line, "@string") ? "string" :
                    (strpos($last_line, "@int") ? "int" : "bool")
            ];
        }

        return $pp;
    }

    static function getArg($str, $props) {
	    foreach ($props as $name => $p) {
	        if ($str == $p["long"] || ( $p["short"] != null && $str == $p["short"])) {
	            return $p;
            }
        }
	    return null;
    }

}
