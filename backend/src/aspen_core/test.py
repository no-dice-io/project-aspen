from utils.bing.base import BingClient

async def test_bing_client():
    client = BingClient()

    # Test web search
    query = f"How often does the Federal Reserve meet?"
    response = await client.async_search(
        api="web_search"
        , query=query
    )

    print(response)

    assert response.status == 200
    assert response.json()


import asyncio
asyncio.run(test_bing_client())