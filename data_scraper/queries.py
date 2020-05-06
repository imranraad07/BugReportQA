from string import Template

repo_query = Template("""
{
  rateLimit {
    cost
    remaining
    resetAt
  }
  search(query: "is:public archived:false pushed:>2019-01-01 created:$start..$end", after: $cursorStart, type: REPOSITORY, first: 80) {
    repositoryCount
    pageInfo {
      hasNextPage
      endCursor
    }
    edges {
      node {
        ... on Repository {
          url
          name
          createdAt
          pushedAt
          issues(first: 1, after: null) {
            totalCount
          }
        }
      }
    }
  }
}
""")

edit_query = Template("""
{
  rateLimit {
    cost
    remaining
    resetAt
  }
  repository(owner: $owner, name: $name) {
    issue(number: $number) {
      number
      userContentEdits(first: 100) {
        totalCount
        edges {
          node {
            diff
          }
        }
      }
    }
  }
}
""")

issues_query = Template("""
{
  rateLimit {
    cost
    remaining
    resetAt
  }
  repository(owner: $owner, name: $name) {
    issues(first: 100, after:$cursor) {
      totalCount
      pageInfo {
        hasNextPage
        endCursor
      }
      edges {
        node {
          ... on Issue {
            number
            author {
              login
            }
          }
        }
      }
    }
  }
}
""")

full_query = """
{
  rateLimit {
    cost
    remaining
    resetAt
  }
  search(query: "is:public archived:false pushed:>2019-01-01", after: null, type: REPOSITORY, first: 20) {
    repositoryCount
    pageInfo {
      hasNextPage
      startCursor
      endCursor
    }
    edges {
      node {
        ... on Repository {
          url
          name
          createdAt
          issues(first: 10, after: null, filterBy: {labels: ["bug", "problem", "Bug", "BUG"]}) {
            totalCount
            pageInfo {
              hasNextPage
              startCursor
            }
            nodes {
              id
              author {
                login
              }
            }
          }
          defaultBranchRef {
            target {
              ... on Commit {
                history(first: 1) {
                  totalCount
                  edges {
                    node {
                      ... on Commit {
                        committedDate
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
"""
