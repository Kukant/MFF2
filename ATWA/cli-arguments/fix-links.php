<?php

/*
 * A script that reads a XHTML file and changes absolute URLs to relative URLs according to given base-URL.
 * This exercise covers mainly OOP, argument processing, and an example in simple XML manipulation.
 */

require_once(__DIR__ . '/args.php');


/**
 * Fix URLs in given XML node and its subtree.
 * @param $xml SimpleXMLElement representing the root of the subtree.
 * @param $base String with the base used for fixing URLs.
 * @param $length Minimal fixing length (shorter URLs are not fixed). If null, no length test is performed.
 * @param $urlAttributes Array [ element => attribute ] of all elements-attributes that needs to be fixed.
 */
function fixUrl($xml, $base, $length, array $urlAttributes)
{
	// Recursively fix rest whole subtree ...
	foreach ($xml->children() as $child)
		fixUrl($child, $base, $length, $urlAttributes);

	// Fix this node ...
	$name = $xml->getName();
	if (empty($urlAttributes[$name])) return;

	// Current node is one of the nodes with URL attributes ...
	$attrName = $urlAttributes[$name];
	if ($xml[$attrName] === null) return;

	// Perform checks, that this URL needs fixing ...
	$url = (string)$xml[$attrName];
	if ($length !== null && strlen($url) <= $length) return;
	if (strlen($url) < strlen($base) || substr($url, 0, strlen($base)) != $base) return;

	// Remove the base from the URL ...
	$xml[$attrName] = substr($url, strlen($base));
}


/*
 * Main execution part of the script...
 */
function main(array $argv) {
	/**
	 * List of all HTML attributes, which could possibly contain URL. Key is element name, value is attribute name.
	 * Note the list is not complete, it contains only some elements to simplify the situation.
	 */
	$urlAttributes = [
		'a' => 'href',
		'img' => 'src',
		'form' => 'action',
		'iframe' => 'src',
		'link' => 'href',
		'script' => 'src',
	];
	
	
	try {
		list($inFileName, $outFileName) = Args::load($argv);
		if (!$inFileName || !file_exists($inFileName))
			throw new Exception("Input file '$inFileName' not found.");
		if (!$outFileName)
			throw new Exception("Output file not specified.");
	
		// Load XML file ...
		echo "Loading file '$inFileName' ...\n";
		libxml_use_internal_errors(true);
		$xmlRoot = simplexml_load_file($inFileName);
		if ($xmlRoot === false) {
			echo "Failed loading file '$inFileName' ...\n";
			foreach(libxml_get_errors() as $error) {
				echo "\t", $error->message;
			}
			exit(1);
		}
	
		// Make sure only the selected attributes are fixed.
		if (Args::$no_imgs) unset($urlAttributes['img']);
		if (Args::$no_links) unset($urlAttributes['a']);
		if (Args::$no_others) $urlAttributes = array_filter($urlAttributes, function($e){
			return ($e == 'a' || $e == 'img');	// keep only <a> and <img>
		}, ARRAY_FILTER_USE_KEY);
	
		if (!$urlAttributes) {
			echo "Configuration says there is nothing to fix ...\n";
			exit;
		}
	
		// Proceed with the operation ...
		echo "Fixing URLs ...\n";
		fixUrl($xmlRoot, Args::$base, Args::$length, $urlAttributes);

		echo "Saving file '$outFileName' ...\n";
		$xmlRoot->asXML($outFileName);

	}
	catch (Exception $e) {
		echo "Error: ", $e->getMessage();
		exit(1);
	}
}

main($argv);
