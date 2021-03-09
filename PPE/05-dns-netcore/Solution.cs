using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Data.Common;
using System.Diagnostics;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace dns_netcore
{
	/*
	 * serial results
	Domain www.ksi.ms.mff.cuni.cz has IP 10.0.0.7 (elapsed time 3022 ms) 
	Domain ksi.ms.mff.cuni.cz has IP 10.0.0.6 (elapsed time 2521 ms) 
	Domain ms.mff.cuni.cz has IP 10.0.0.5 (elapsed time 2019 ms) 
	Domain mff.cuni.cz has IP 10.0.0.4 (elapsed time 1516 ms) 
	Domain cuni.cz has IP 10.0.0.3 (elapsed time 1011 ms) 
	Domain cz has IP 10.0.0.2 (elapsed time 507 ms) 
	Starting ... 3 tests
	Domain parlab.ms.mff.cuni.cz has IP 10.0.0.8 (elapsed time 2523 ms) 
	Domain www.seznam.cz has IP 10.0.0.10 (elapsed time 1514 ms) 
	Domain www.google.com has IP 10.0.0.13 (elapsed time 1514 ms) 
	 */
	class RecursiveResolver : IRecursiveResolver
	{
		private IDNSClient dnsClient;
		
		private ConcurrentDictionary<String, Task<IP4Addr>> IpCache;
		private static Mutex mut = new Mutex();
		
		private IP4Addr resolveSubDomain(string domain, IP4Addr start) {
			string[] domains = domain.Split('.');
			Array.Reverse(domains);
			IP4Addr res = start;
			
			for (var i = 0; i < domains.Length; i++) {
				var sub = domains[i];
				
				string cacheKey = "";
				for (var j = i; j >= 0; j--) {
					cacheKey += domains[j] + (j == 0 ? "" : ".");
				}

				Task<IP4Addr> t;

				mut.WaitOne();
				if (IpCache.ContainsKey(cacheKey)) {
					t = IpCache[cacheKey];
					dbg("Cache hit " + cacheKey);
				} else {
					dbg("Setting cache for " + cacheKey);
					t = dnsClient.Resolve(res, sub);
					IpCache[cacheKey] = t;
				}
				mut.ReleaseMutex();

				if (t.IsCompleted) {
					dbg("Cache needs verifying");
					// task has already been finished, verify cache
					Task<String> reverse = dnsClient.Reverse(t.Result);
					reverse.Wait();
					if (reverse.Result != cacheKey) {
						dbg("Cache verification failed. " + reverse.Result);
						IpCache.Remove(cacheKey, out Task<IP4Addr> v);
						// go level back
						i = Math.Max(-1, i - 2);
						continue;
					}
				}
				
				t.Wait();
				res = t.Result;
			}

			return res;
		}

		private void dbg(String msg) {
			bool debug = false;
			if (debug) {
				Console.WriteLine(msg);
			}
		}

		private IP4Addr ResolveWithCache(string domain) {
			IP4Addr res = dnsClient.GetRootServers()[0];
			return resolveSubDomain(domain, res);
		}

		public RecursiveResolver(IDNSClient client)
		{
			dnsClient = client;
			IpCache = new ConcurrentDictionary<string, Task<IP4Addr>>();
		}

		public Task<IP4Addr> ResolveRecursive(string domain)
		{
			return Task<IP4Addr>.Run(() => {
				dbg("Resolving " + domain);
				return ResolveWithCache(domain);
			});
		}
	}
}
